use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use jwalk::{Parallelism, WalkDir};
use pathdiff::diff_paths;
use std::collections::VecDeque;
use std::env;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt, symlink};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[cfg(target_env = "musl")]
use tikv_jemallocator::Jemalloc;

#[cfg(target_env = "musl")]
#[global_allocator]
static GLOBAL_ALLOCATOR: Jemalloc = Jemalloc;

const MIN_CHUNK_SIZE: usize = 32 * 1024 * 1024;
const FALLBACK_IO_CHUNK_SIZE: usize = 8 * 1024 * 1024;
const MIN_SINGLE_FILE_RANGE_SIZE: usize = 4 * 1024 * 1024;
const MAX_WORKER_THREADS: usize = 32;

#[derive(Debug, Parser)]
#[command(
    name = "cpb",
    about = "Parallel copy with reflink support",
    override_usage = "cpb [OPTIONS] <SOURCE>... <DESTINATION>",
    after_help = "Arguments:\n  <SOURCE>...   One or more source paths\n  <DESTINATION> Destination path (when using multiple sources, must be an existing directory)\n\nExamples:\n  cpb file1 file2 dir1\n  cpb source.txt destination.txt\n  cpb src_dir backup_root"
)]
struct Args {
    #[arg(
        required = true,
        num_args = 2..,
        value_name = "PATH",
        help = "Paths where the last value is DESTINATION and preceding values are SOURCE(s)"
    )]
    paths: Vec<PathBuf>,
    #[arg(long, default_value_t = false)]
    silent: bool,
}

#[derive(Debug, Clone, Copy)]
struct PreservedMetadata {
    mode: u32,
    uid: u32,
    gid: u32,
    atime_sec: i64,
    atime_nsec: i64,
    mtime_sec: i64,
    mtime_nsec: i64,
}

#[derive(Debug, Clone)]
struct FileJob {
    source: PathBuf,
    destination: PathBuf,
    size: u64,
    metadata: PreservedMetadata,
}

#[derive(Debug, Clone)]
struct SymlinkJob {
    destination: PathBuf,
    target: PathBuf,
    metadata: PreservedMetadata,
}

#[derive(Debug, Default)]
struct CopyPlan {
    file_jobs: Vec<FileJob>,
    symlink_jobs: Vec<SymlinkJob>,
    directory_metadata: Vec<(PathBuf, PreservedMetadata)>,
}

#[derive(Debug, Clone, Copy)]
enum WorkItem {
    File(usize),
    Symlink(usize),
}

fn default_worker_threads() -> usize {
    thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, MAX_WORKER_THREADS)
}

fn split_sources_and_destination(paths: &[PathBuf]) -> io::Result<(&[PathBuf], &Path)> {
    if paths.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at least one source and one destination path are required",
        ));
    }

    let destination = paths
        .last()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "destination path missing"))?;
    let sources = &paths[..paths.len() - 1];
    Ok((sources, destination.as_path()))
}

fn preserved_metadata_from(metadata: &fs::Metadata) -> PreservedMetadata {
    PreservedMetadata {
        mode: metadata.permissions().mode(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        atime_sec: metadata.atime(),
        atime_nsec: metadata.atime_nsec(),
        mtime_sec: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
    }
}

fn resolve_file_destination(input: &Path, output: &Path) -> io::Result<PathBuf> {
    match fs::metadata(output) {
        Ok(metadata) if metadata.is_dir() => {
            let file_name = input.file_name().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "input file name is missing")
            })?;
            Ok(output.join(file_name))
        }
        Ok(_) => Ok(output.to_path_buf()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(output.to_path_buf()),
        Err(error) => Err(error),
    }
}

fn resolve_directory_destination(input: &Path, output: &Path) -> io::Result<PathBuf> {
    match fs::metadata(output) {
        Ok(metadata) => {
            if !metadata.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "output must be a directory when input is a directory",
                ));
            }

            let directory_name = input.file_name().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "input directory name is missing",
                )
            })?;
            Ok(output.join(directory_name))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(output.to_path_buf()),
        Err(error) => Err(error),
    }
}

fn is_path_within_base(candidate: &Path, base: &Path) -> bool {
    match diff_paths(candidate, base) {
        Some(relative) if relative.as_os_str().is_empty() => true,
        Some(relative) => !matches!(relative.components().next(), Some(Component::ParentDir)),
        None => false,
    }
}

fn ensure_destination_outside_source(source: &Path, destination_root: &Path) -> io::Result<()> {
    let source_canonical = fs::canonicalize(source)?;

    let destination_approx = if destination_root.exists() {
        fs::canonicalize(destination_root)?
    } else if destination_root.is_absolute() {
        destination_root.to_path_buf()
    } else {
        env::current_dir()?.join(destination_root)
    };

    if is_path_within_base(&destination_approx, &source_canonical) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot copy a directory into itself",
        ));
    }

    Ok(())
}

fn relative_path_from(path: &Path, base: &Path) -> io::Result<PathBuf> {
    if path == base {
        return Ok(PathBuf::new());
    }

    diff_paths(path, base).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "cannot compute relative path from {} to {}",
                path.display(),
                base.display()
            ),
        )
    })
}

fn collect_directory_jobs(
    source_directory: &Path,
    destination_directory: &Path,
    walk_threads: usize,
    plan: &mut CopyPlan,
) -> io::Result<()> {
    for entry_result in WalkDir::new(source_directory)
        .skip_hidden(false)
        .follow_links(false)
        .parallelism(Parallelism::RayonNewPool(walk_threads.max(1)))
    {
        let entry = match entry_result {
            Ok(entry) => entry,
            Err(error) => return Err(io::Error::other(error.to_string())),
        };
        let source_path = entry.path();
        let relative_path = relative_path_from(&source_path, source_directory)?;
        let destination_path = if relative_path.as_os_str().is_empty() {
            destination_directory.to_path_buf()
        } else {
            destination_directory.join(relative_path)
        };

        let metadata = fs::symlink_metadata(&source_path)?;
        let file_type = metadata.file_type();

        if file_type.is_dir() {
            plan.directory_metadata
                .push((destination_path, preserved_metadata_from(&metadata)));
            continue;
        }

        if file_type.is_file() {
            plan.file_jobs.push(FileJob {
                source: source_path,
                destination: destination_path,
                size: metadata.len(),
                metadata: preserved_metadata_from(&metadata),
            });
            continue;
        }

        if file_type.is_symlink() {
            let target = fs::read_link(&source_path)?;

            plan.symlink_jobs.push(SymlinkJob {
                destination: destination_path,
                target,
                metadata: preserved_metadata_from(&metadata),
            });
            continue;
        }

        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unsupported file type encountered at {}",
                source_path.display()
            ),
        ));
    }

    Ok(())
}

fn build_copy_plan(input: &Path, output: &Path, walk_threads: usize) -> io::Result<CopyPlan> {
    let source_metadata = fs::symlink_metadata(input)?;
    let source_type = source_metadata.file_type();

    if source_type.is_file() {
        let destination = resolve_file_destination(input, output)?;

        if let Ok(destination_metadata) = fs::metadata(&destination)
            && source_metadata.dev() == destination_metadata.dev()
            && source_metadata.ino() == destination_metadata.ino()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input and output point to the same file",
            ));
        }

        return Ok(CopyPlan {
            file_jobs: vec![FileJob {
                source: input.to_path_buf(),
                destination,
                size: source_metadata.len(),
                metadata: preserved_metadata_from(&source_metadata),
            }],
            symlink_jobs: Vec::new(),
            directory_metadata: Vec::new(),
        });
    }

    if source_type.is_dir() {
        let destination_root = resolve_directory_destination(input, output)?;
        ensure_destination_outside_source(input, &destination_root)?;

        let mut plan = CopyPlan::default();
        collect_directory_jobs(input, &destination_root, walk_threads, &mut plan)?;
        return Ok(plan);
    }

    if source_type.is_symlink() {
        let destination = resolve_file_destination(input, output)?;

        if let Ok(destination_metadata) = fs::symlink_metadata(&destination)
            && source_metadata.dev() == destination_metadata.dev()
            && source_metadata.ino() == destination_metadata.ino()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "input and output point to the same file",
            ));
        }

        let target = fs::read_link(input)?;
        return Ok(CopyPlan {
            file_jobs: Vec::new(),
            symlink_jobs: vec![SymlinkJob {
                destination,
                target,
                metadata: preserved_metadata_from(&source_metadata),
            }],
            directory_metadata: Vec::new(),
        });
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "input must be a regular file, symlink, or directory",
    ))
}

fn build_copy_plan_for_sources(
    sources: &[PathBuf],
    output: &Path,
    walk_threads: usize,
) -> io::Result<CopyPlan> {
    if sources.len() > 1 {
        match fs::metadata(output) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) | Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "destination must be an existing directory when copying multiple sources",
                ));
            }
        }
    }

    let mut merged_plan = CopyPlan::default();
    for source in sources {
        let mut source_plan = build_copy_plan(source, output, walk_threads)?;
        merged_plan.file_jobs.append(&mut source_plan.file_jobs);
        merged_plan
            .symlink_jobs
            .append(&mut source_plan.symlink_jobs);
        merged_plan
            .directory_metadata
            .append(&mut source_plan.directory_metadata);
    }

    Ok(merged_plan)
}

fn split_ranges(total_size: u64, workers: usize) -> Vec<(u64, u64)> {
    let mut ranges = Vec::with_capacity(workers);
    for index in 0..workers {
        let start = total_size.saturating_mul(index as u64) / workers as u64;
        let end = total_size.saturating_mul(index as u64 + 1) / workers as u64;
        if end > start {
            ranges.push((start, end - start));
        }
    }
    ranges
}

fn single_file_worker_count(file_size: u64, requested_threads: usize) -> usize {
    let max_useful_workers = file_size.div_ceil(MIN_SINGLE_FILE_RANGE_SIZE as u64);
    let max_useful_workers = usize::try_from(max_useful_workers).unwrap_or(requested_threads);
    requested_threads.min(max_useful_workers.max(1))
}

fn prepare_destination_layout(plan: &CopyPlan) -> io::Result<()> {
    for (directory, _) in &plan.directory_metadata {
        fs::create_dir_all(directory)?;
    }

    for job in &plan.file_jobs {
        if let Some(parent) = job.destination.parent() {
            fs::create_dir_all(parent)?;
        }
    }

    for job in &plan.symlink_jobs {
        if let Some(parent) = job.destination.parent() {
            fs::create_dir_all(parent)?;
        }
    }

    Ok(())
}

fn should_fallback_copy_file_range(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EXDEV | libc::EOPNOTSUPP | libc::ENOSYS | libc::EINVAL)
    )
}

fn remove_existing_non_directory_destination(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "destination path {} is an existing directory",
                path.display()
            ),
        )),
        Ok(_) => fs::remove_file(path),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn prepare_destination_file(path: &Path, size: u64) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .read(true)
        .open(path)?;
    output.set_len(size)?;
    Ok(())
}

fn try_reflink(job: &FileJob, progress: &ProgressBar) -> io::Result<bool> {
    remove_existing_non_directory_destination(&job.destination)?;

    match reflink_copy::reflink(&job.source, &job.destination) {
        Ok(()) => {
            progress.inc(job.size);
            Ok(true)
        }
        Err(_) => {
            match fs::remove_file(&job.destination) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            Ok(false)
        }
    }
}

fn copy_range_with_read_write(
    source: &File,
    destination: &File,
    start: u64,
    len: u64,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    let io_chunk_size = chunk_size.clamp(64 * 1024, FALLBACK_IO_CHUNK_SIZE);
    let mut offset = start;
    let mut remaining = len;
    let mut buffer = vec![0_u8; io_chunk_size];

    while remaining > 0 {
        let read_len = remaining.min(io_chunk_size as u64) as usize;
        let bytes_read = loop {
            match source.read_at(&mut buffer[..read_len], offset) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "source ended before copy completed",
                    ));
                }
                Ok(read) => break read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        };

        let mut written = 0_usize;
        while written < bytes_read {
            match destination.write_at(&buffer[written..bytes_read], offset + written as u64) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write copied bytes",
                    ));
                }
                Ok(count) => written += count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }

        let copied = written as u64;
        remaining -= copied;
        offset += copied;
        progress.inc(copied);
    }

    Ok(())
}

fn copy_range_with_syscall<F>(
    input: &Path,
    output: &Path,
    start: u64,
    len: u64,
    chunk_size: usize,
    progress: &ProgressBar,
    mut copy_syscall: F,
) -> io::Result<()>
where
    F: FnMut(libc::c_int, &mut libc::loff_t, libc::c_int, &mut libc::loff_t, usize) -> isize,
{
    let src = OpenOptions::new().read(true).open(input)?;
    let dst = OpenOptions::new().write(true).open(output)?;

    let src_fd = src.as_raw_fd();
    let dst_fd = dst.as_raw_fd();

    let mut in_offset = i64::try_from(start).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "source offset exceeds supported range",
        )
    })?;
    let mut out_offset = in_offset;

    let mut remaining = len;
    while remaining > 0 {
        let to_copy = remaining.min(chunk_size as u64) as usize;
        let copied = copy_syscall(src_fd, &mut in_offset, dst_fd, &mut out_offset, to_copy);

        if copied < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }

            if should_fallback_copy_file_range(&error) {
                let offset = in_offset as u64;
                return copy_range_with_read_write(
                    &src, &dst, offset, remaining, chunk_size, progress,
                );
            }

            return Err(error);
        }

        if copied == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "copy_file_range returned 0 before completion",
            ));
        }

        let copied = copied as u64;
        remaining -= copied;
        progress.inc(copied);
    }

    Ok(())
}

fn copy_range(
    input: &Path,
    output: &Path,
    start: u64,
    len: u64,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    copy_range_with_syscall(
        input,
        output,
        start,
        len,
        chunk_size,
        progress,
        |src_fd, in_offset, dst_fd, out_offset, to_copy| unsafe {
            libc::copy_file_range(
                src_fd,
                in_offset as *mut libc::loff_t,
                dst_fd,
                out_offset as *mut libc::loff_t,
                to_copy,
                0,
            )
        },
    )
}

fn copy_file_job(job: &FileJob, chunk_size: usize, progress: &ProgressBar) -> io::Result<()> {
    if job.size == 0 {
        remove_existing_non_directory_destination(&job.destination)?;
        prepare_destination_file(&job.destination, 0)?;
        return Ok(());
    }

    if try_reflink(job, progress)? {
        return Ok(());
    }

    prepare_destination_file(&job.destination, job.size)?;
    copy_range(
        &job.source,
        &job.destination,
        0,
        job.size,
        chunk_size,
        progress,
    )
}

fn join_io_worker(handle: thread::JoinHandle<io::Result<()>>) -> io::Result<()> {
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(io::Error::other("copy worker panicked")),
    }
}

fn join_unit_worker(handle: thread::JoinHandle<()>) -> io::Result<()> {
    if handle.join().is_err() {
        return Err(io::Error::other("copy worker panicked"));
    }
    Ok(())
}

fn copy_single_file_with_ranges(
    job: &FileJob,
    threads: usize,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    if job.size == 0 {
        remove_existing_non_directory_destination(&job.destination)?;
        prepare_destination_file(&job.destination, 0)?;
        return Ok(());
    }

    if try_reflink(job, progress)? {
        return Ok(());
    }

    prepare_destination_file(&job.destination, job.size)?;

    let worker_count = single_file_worker_count(job.size, threads);
    let ranges = split_ranges(job.size, worker_count);

    let mut handles = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let source = job.source.clone();
        let destination = job.destination.clone();
        let thread_progress = progress.clone();

        handles.push(thread::spawn(move || {
            copy_range(
                &source,
                &destination,
                start,
                len,
                chunk_size,
                &thread_progress,
            )
        }));
    }

    for handle in handles {
        join_io_worker(handle)?;
    }

    Ok(())
}

fn copy_symlink_job(job: &SymlinkJob) -> io::Result<()> {
    remove_existing_non_directory_destination(&job.destination)?;
    symlink(&job.target, &job.destination)
}

fn copy_jobs_with_pool(
    file_jobs: &[FileJob],
    symlink_jobs: &[SymlinkJob],
    threads: usize,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    let total_jobs = file_jobs.len() + symlink_jobs.len();
    if total_jobs == 0 {
        return Ok(());
    }

    let worker_count = threads.min(total_jobs).max(1);
    let mut work_queue = VecDeque::with_capacity(total_jobs);
    for index in 0..file_jobs.len() {
        work_queue.push_back(WorkItem::File(index));
    }
    for index in 0..symlink_jobs.len() {
        work_queue.push_back(WorkItem::Symlink(index));
    }

    let queue = Arc::new(Mutex::new(work_queue));
    let shared_file_jobs = Arc::new(file_jobs.to_vec());
    let shared_symlink_jobs = Arc::new(symlink_jobs.to_vec());
    let has_failed = Arc::new(AtomicBool::new(false));
    let first_error = Arc::new(Mutex::new(None::<io::Error>));

    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let shared_file_jobs = Arc::clone(&shared_file_jobs);
        let shared_symlink_jobs = Arc::clone(&shared_symlink_jobs);
        let has_failed = Arc::clone(&has_failed);
        let first_error = Arc::clone(&first_error);
        let thread_progress = progress.clone();

        handles.push(thread::spawn(move || {
            loop {
                // Relaxed ordering is sufficient here: this flag is only a best-effort
                // early-exit signal, and workers can safely process a little extra work.
                if has_failed.load(Ordering::Relaxed) {
                    break;
                }

                let next_job = {
                    let mut queue = queue.lock().expect("copy queue mutex poisoned");
                    queue.pop_front()
                };

                let Some(next_job) = next_job else {
                    break;
                };

                let job_result = match next_job {
                    WorkItem::File(index) => {
                        copy_file_job(&shared_file_jobs[index], chunk_size, &thread_progress)
                    }
                    WorkItem::Symlink(index) => copy_symlink_job(&shared_symlink_jobs[index]),
                };

                if let Err(error) = job_result {
                    has_failed.store(true, Ordering::Relaxed);
                    let mut slot = first_error.lock().expect("error slot mutex poisoned");
                    if slot.is_none() {
                        *slot = Some(error);
                    }
                    break;
                }
            }
        }));
    }

    for handle in handles {
        join_unit_worker(handle)?;
    }

    let mut slot = first_error.lock().expect("error slot mutex poisoned");
    if let Some(error) = slot.take() {
        return Err(error);
    }

    Ok(())
}

fn path_to_c_string(path: &Path) -> io::Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path contains NUL byte: {}", path.display()),
        )
    })
}

fn should_ignore_ownership_error(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EPERM | libc::EACCES | libc::ENOSYS | libc::EOPNOTSUPP)
    )
}

fn set_path_owner(
    path: &Path,
    metadata: PreservedMetadata,
    follow_symlink: bool,
) -> io::Result<()> {
    let path = path_to_c_string(path)?;
    let flags = if follow_symlink {
        0
    } else {
        libc::AT_SYMLINK_NOFOLLOW
    };

    let result = unsafe {
        libc::fchownat(
            libc::AT_FDCWD,
            path.as_ptr(),
            metadata.uid,
            metadata.gid,
            flags,
        )
    };

    if result == 0 {
        return Ok(());
    }

    let error = io::Error::last_os_error();
    if should_ignore_ownership_error(&error) {
        return Ok(());
    }

    Err(error)
}

fn set_path_timestamps(
    path: &Path,
    metadata: PreservedMetadata,
    follow_symlink: bool,
) -> io::Result<()> {
    let path = path_to_c_string(path)?;
    let flags = if follow_symlink {
        0
    } else {
        libc::AT_SYMLINK_NOFOLLOW
    };
    let times = [
        libc::timespec {
            tv_sec: metadata.atime_sec as libc::time_t,
            tv_nsec: metadata.atime_nsec as libc::c_long,
        },
        libc::timespec {
            tv_sec: metadata.mtime_sec as libc::time_t,
            tv_nsec: metadata.mtime_nsec as libc::c_long,
        },
    ];

    let result = unsafe { libc::utimensat(libc::AT_FDCWD, path.as_ptr(), times.as_ptr(), flags) };
    if result == 0 {
        return Ok(());
    }

    Err(io::Error::last_os_error())
}

fn apply_preserved_metadata(plan: &CopyPlan) -> io::Result<()> {
    for job in &plan.file_jobs {
        set_path_owner(&job.destination, job.metadata, true)?;
        fs::set_permissions(
            &job.destination,
            fs::Permissions::from_mode(job.metadata.mode),
        )?;
        set_path_timestamps(&job.destination, job.metadata, true)?;
    }

    for job in &plan.symlink_jobs {
        set_path_owner(&job.destination, job.metadata, false)?;
        set_path_timestamps(&job.destination, job.metadata, false)?;
    }

    for (directory, metadata) in plan.directory_metadata.iter().rev() {
        set_path_owner(directory, *metadata, true)?;
        fs::set_permissions(directory, fs::Permissions::from_mode(metadata.mode))?;
        set_path_timestamps(directory, *metadata, true)?;
    }

    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (sources, destination) = split_sources_and_destination(&args.paths)?;
    let worker_threads = default_worker_threads();
    let chunk_size = MIN_CHUNK_SIZE;
    let plan = build_copy_plan_for_sources(sources, destination, worker_threads)?;

    let total_size = plan.file_jobs.iter().map(|job| job.size).sum::<u64>();
    let total_files = plan.file_jobs.len();
    let total_symlinks = plan.symlink_jobs.len();

    let progress = if args.silent {
        ProgressBar::hidden()
    } else {
        let progress = ProgressBar::new(total_size);
        let style = ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta}) {msg}",
        )
        .expect("hardcoded progress template is valid")
        .progress_chars("=> ");
        progress.set_style(style);
        progress.enable_steady_tick(Duration::from_millis(120));
        progress.set_message("preparing destination");
        progress.tick();
        progress
    };

    prepare_destination_layout(&plan)?;

    if !args.silent {
        progress.set_message("copying");
    }

    let copy_result = if total_files == 1 && total_symlinks == 0 {
        copy_single_file_with_ranges(&plan.file_jobs[0], worker_threads, chunk_size, &progress)
    } else {
        copy_jobs_with_pool(
            &plan.file_jobs,
            &plan.symlink_jobs,
            worker_threads,
            chunk_size,
            &progress,
        )
    };

    if let Err(error) = copy_result {
        if !args.silent {
            progress.abandon();
        }
        return Err(error.into());
    }

    apply_preserved_metadata(&plan)?;
    if !args.silent {
        progress.finish_with_message(format!(
            "copied {total_size} bytes across {total_files} file(s) and {total_symlinks} symlink(s)"
        ));
    }

    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let silent = args.silent;
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if !silent {
                eprintln!("Error: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::io::Write;
    use std::os::unix::ffi::OsStrExt;
    use tempfile::tempdir;

    fn write_file_with_mode(path: &Path, data: &[u8], mode: u32) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        file.write_all(data)?;
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
        Ok(())
    }

    fn patterned_data(size: usize) -> Vec<u8> {
        (0..size).map(|index| (index % 251) as u8).collect()
    }

    fn file_job_from_paths(source: &Path, destination: &Path) -> FileJob {
        let metadata = fs::symlink_metadata(source).expect("failed to read source metadata");
        FileJob {
            source: source.to_path_buf(),
            destination: destination.to_path_buf(),
            size: metadata.len(),
            metadata: preserved_metadata_from(&metadata),
        }
    }

    #[test]
    fn default_worker_threads_is_within_expected_bounds() {
        let threads = default_worker_threads();
        assert!((1..=MAX_WORKER_THREADS).contains(&threads));
    }

    #[test]
    fn split_sources_and_destination_requires_at_least_two_paths() {
        let error =
            split_sources_and_destination(&[]).expect_err("splitting without paths should fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        let source = PathBuf::from("source");
        let destination = PathBuf::from("destination");
        let paths = vec![source.clone(), destination.clone()];
        let (sources, resolved_destination) = split_sources_and_destination(&paths)
            .expect("splitting source and destination should succeed");

        assert_eq!(sources, &[source]);
        assert_eq!(resolved_destination, destination);
    }

    #[test]
    fn resolve_file_destination_covers_all_cases() {
        let temp = tempdir().expect("failed to create tempdir");
        let input = temp.path().join("source.txt");
        write_file_with_mode(&input, b"content", 0o644).expect("failed to write input file");

        let output_directory = temp.path().join("output-dir");
        fs::create_dir(&output_directory).expect("failed to create output directory");
        assert_eq!(
            resolve_file_destination(&input, &output_directory)
                .expect("directory output resolution should succeed"),
            output_directory.join("source.txt")
        );

        let existing_file_output = temp.path().join("existing.txt");
        write_file_with_mode(&existing_file_output, b"old", 0o644)
            .expect("failed to write existing output file");
        assert_eq!(
            resolve_file_destination(&input, &existing_file_output)
                .expect("existing file output resolution should succeed"),
            existing_file_output
        );

        let new_file_output = temp.path().join("new.txt");
        assert_eq!(
            resolve_file_destination(&input, &new_file_output)
                .expect("new file output resolution should succeed"),
            new_file_output
        );

        let missing_name_error = resolve_file_destination(Path::new("/"), &output_directory)
            .expect_err("expected missing name error");
        assert_eq!(missing_name_error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn resolve_directory_destination_covers_all_cases() {
        let temp = tempdir().expect("failed to create tempdir");
        let input_directory = temp.path().join("source-dir");
        fs::create_dir(&input_directory).expect("failed to create input directory");

        let output_directory = temp.path().join("output-dir");
        fs::create_dir(&output_directory).expect("failed to create output directory");
        assert_eq!(
            resolve_directory_destination(&input_directory, &output_directory)
                .expect("directory destination resolution should succeed"),
            output_directory.join("source-dir")
        );

        let existing_file_output = temp.path().join("existing.txt");
        write_file_with_mode(&existing_file_output, b"old", 0o644)
            .expect("failed to write existing output file");
        let error = resolve_directory_destination(&input_directory, &existing_file_output)
            .expect_err("expected error for file output");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        let new_directory_output = temp.path().join("new-output-dir");
        assert_eq!(
            resolve_directory_destination(&input_directory, &new_directory_output)
                .expect("new directory output resolution should succeed"),
            new_directory_output
        );
    }

    #[test]
    fn is_path_within_base_detects_relationships() {
        let base = Path::new("/tmp/base");
        assert!(is_path_within_base(Path::new("/tmp/base"), base));
        assert!(is_path_within_base(Path::new("/tmp/base/nested"), base));
        assert!(!is_path_within_base(Path::new("/tmp/other"), base));
    }

    #[test]
    fn ensure_destination_outside_source_rejects_nested_paths() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("src");
        fs::create_dir(&source).expect("failed to create source directory");

        let outside_destination = temp.path().join("dst");
        ensure_destination_outside_source(&source, &outside_destination)
            .expect("outside destination should be accepted");

        let nested_destination = source.join("nested");
        let error = ensure_destination_outside_source(&source, &nested_destination)
            .expect_err("nested destination should be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        let escaped_destination = source.join("..").join("escaped");
        ensure_destination_outside_source(&source, &escaped_destination)
            .expect("escaped destination should be accepted");
    }

    #[test]
    fn relative_path_from_builds_expected_paths() {
        let base = Path::new("/tmp/base");
        let nested = Path::new("/tmp/base/a/b");
        assert_eq!(
            relative_path_from(nested, base).expect("nested relative path should resolve"),
            PathBuf::from("a/b")
        );
        assert_eq!(
            relative_path_from(base, base).expect("base relative path should resolve"),
            PathBuf::new()
        );

        let error = relative_path_from(Path::new("relative"), Path::new("/absolute"))
            .expect_err("mismatched path styles should fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn split_ranges_evenly_covers_all_bytes() {
        assert_eq!(split_ranges(10, 3), vec![(0, 3), (3, 3), (6, 4)]);
        assert_eq!(split_ranges(0, 4), Vec::<(u64, u64)>::new());
    }

    #[test]
    fn single_file_worker_count_caps_worker_usage() {
        assert_eq!(single_file_worker_count(0, 8), 1);
        assert_eq!(
            single_file_worker_count((MIN_SINGLE_FILE_RANGE_SIZE * 2) as u64, 8),
            2
        );
        assert_eq!(
            single_file_worker_count((MIN_SINGLE_FILE_RANGE_SIZE * 32) as u64, 4),
            4
        );
    }

    #[test]
    fn should_fallback_copy_file_range_matches_expected_errno_values() {
        assert!(should_fallback_copy_file_range(
            &io::Error::from_raw_os_error(libc::EXDEV)
        ));
        assert!(should_fallback_copy_file_range(
            &io::Error::from_raw_os_error(libc::EOPNOTSUPP)
        ));
        assert!(should_fallback_copy_file_range(
            &io::Error::from_raw_os_error(libc::ENOSYS)
        ));
        assert!(should_fallback_copy_file_range(
            &io::Error::from_raw_os_error(libc::EINVAL)
        ));
        assert!(!should_fallback_copy_file_range(
            &io::Error::from_raw_os_error(libc::EACCES)
        ));
    }

    #[test]
    fn should_ignore_ownership_error_matches_expected_errno_values() {
        assert!(should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::EPERM)
        ));
        assert!(should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::EACCES)
        ));
        assert!(should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::ENOSYS)
        ));
        assert!(should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::ENOTSUP)
        ));
        assert!(should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::EOPNOTSUPP)
        ));
        assert!(!should_ignore_ownership_error(
            &io::Error::from_raw_os_error(libc::ENOENT)
        ));
    }

    #[test]
    fn build_copy_plan_for_directory_tracks_files_and_symlinks() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source");
        fs::create_dir_all(source.join("nested")).expect("failed to create source tree");
        write_file_with_mode(&source.join("root.txt"), b"root", 0o640)
            .expect("failed to write root file");
        let root_path = source.join("root.txt");
        let root_metadata = fs::symlink_metadata(&root_path).expect("failed to read root metadata");
        set_path_timestamps(
            &root_path,
            PreservedMetadata {
                mode: root_metadata.permissions().mode(),
                uid: root_metadata.uid(),
                gid: root_metadata.gid(),
                atime_sec: 1_234_567_890,
                atime_nsec: 123,
                mtime_sec: 1_234_567_891,
                mtime_nsec: 456,
            },
            true,
        )
        .expect("failed to set root timestamps");
        write_file_with_mode(&source.join(".hidden"), b"hidden", 0o600)
            .expect("failed to write hidden file");
        write_file_with_mode(&source.join("nested/inner.txt"), b"inner", 0o600)
            .expect("failed to write nested file");
        symlink(Path::new("root.txt"), source.join("link-to-root"))
            .expect("failed to create file symlink");
        symlink(Path::new("nested"), source.join("link-to-dir"))
            .expect("failed to create directory symlink");

        let output_parent = temp.path().join("output-parent");
        fs::create_dir(&output_parent).expect("failed to create output parent directory");

        let plan = build_copy_plan(&source, &output_parent, 4)
            .expect("directory copy plan should be created successfully");

        let destination_root = output_parent.join("source");
        assert!(
            plan.directory_metadata
                .iter()
                .any(|(path, _)| path == &destination_root)
        );
        assert_eq!(plan.file_jobs.len(), 3);
        assert_eq!(plan.symlink_jobs.len(), 2);
        assert!(
            plan.file_jobs
                .iter()
                .any(|job| job.destination == destination_root.join("nested/inner.txt"))
        );
        assert!(plan.symlink_jobs.iter().any(|job| {
            job.destination == destination_root.join("link-to-root")
                && job.target == Path::new("root.txt")
        }));
    }

    #[test]
    fn build_copy_plan_rejects_same_file_source_and_destination() {
        let temp = tempdir().expect("failed to create tempdir");
        let file = temp.path().join("same.txt");
        write_file_with_mode(&file, b"same", 0o644).expect("failed to write file");

        let error =
            build_copy_plan(&file, &file, 2).expect_err("same source and destination must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn build_copy_plan_for_top_level_symlink_creates_symlink_job() {
        let temp = tempdir().expect("failed to create tempdir");
        let target = temp.path().join("target.txt");
        write_file_with_mode(&target, b"target", 0o644).expect("failed to write target file");
        let source_link = temp.path().join("source-link");
        symlink(Path::new("target.txt"), &source_link).expect("failed to create source symlink");

        let output_dir = temp.path().join("output");
        fs::create_dir(&output_dir).expect("failed to create output directory");

        let plan = build_copy_plan(&source_link, &output_dir, 2)
            .expect("symlink copy plan should be created successfully");
        assert_eq!(plan.file_jobs.len(), 0);
        assert_eq!(plan.symlink_jobs.len(), 1);
        assert_eq!(
            plan.symlink_jobs[0].destination,
            output_dir.join("source-link")
        );
        assert_eq!(plan.symlink_jobs[0].target, PathBuf::from("target.txt"));
    }

    #[test]
    fn build_copy_plan_rejects_unsupported_entry_type() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source");
        fs::create_dir(&source).expect("failed to create source directory");

        let fifo = source.join("named-pipe");
        let fifo_c_string =
            CString::new(fifo.as_os_str().as_bytes()).expect("failed to build CString");
        let mkfifo_result = unsafe { libc::mkfifo(fifo_c_string.as_ptr(), 0o644) };
        assert_eq!(mkfifo_result, 0, "mkfifo should succeed");

        let destination = temp.path().join("destination");
        let error = build_copy_plan(&source, &destination, 2)
            .expect_err("unsupported entry should fail planning");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn prepare_destination_layout_creates_required_directories() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.txt");
        write_file_with_mode(&source, b"source", 0o644).expect("failed to write source file");

        let destination_root = temp.path().join("destination");
        let nested_file_destination = destination_root.join("a/b/file.txt");
        let nested_link_destination = destination_root.join("x/y/link");

        let plan = CopyPlan {
            file_jobs: vec![file_job_from_paths(&source, &nested_file_destination)],
            symlink_jobs: vec![SymlinkJob {
                destination: nested_link_destination.clone(),
                target: PathBuf::from("target"),
                metadata: preserved_metadata_from(
                    &fs::symlink_metadata(&source).expect("failed to read source metadata"),
                ),
            }],
            directory_metadata: vec![(
                destination_root.join("explicit"),
                PreservedMetadata {
                    mode: 0o755,
                    uid: 0,
                    gid: 0,
                    atime_sec: 0,
                    atime_nsec: 0,
                    mtime_sec: 0,
                    mtime_nsec: 0,
                },
            )],
        };

        prepare_destination_layout(&plan).expect("destination layout preparation should succeed");
        assert!(destination_root.join("explicit").is_dir());
        assert!(destination_root.join("a/b").is_dir());
        assert!(destination_root.join("x/y").is_dir());
        assert!(!nested_file_destination.exists());
        assert!(!nested_link_destination.exists());
    }

    #[test]
    fn remove_existing_non_directory_destination_handles_files_and_directories() {
        let temp = tempdir().expect("failed to create tempdir");
        let file_path = temp.path().join("file.txt");
        write_file_with_mode(&file_path, b"file", 0o644).expect("failed to write file");
        remove_existing_non_directory_destination(&file_path)
            .expect("existing file should be removable");
        assert!(!file_path.exists());

        let directory_path = temp.path().join("dir");
        fs::create_dir(&directory_path).expect("failed to create directory");
        let error = remove_existing_non_directory_destination(&directory_path)
            .expect_err("existing directory should return an error");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn copy_range_with_read_write_copies_requested_slice() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.bin");
        let data = patterned_data(256 * 1024);
        write_file_with_mode(&source, &data, 0o644).expect("failed to write source file");

        let destination = temp.path().join("destination.bin");
        prepare_destination_file(&destination, data.len() as u64)
            .expect("failed to prepare destination file");

        let source_file = OpenOptions::new()
            .read(true)
            .open(&source)
            .expect("failed to open source file");
        let destination_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&destination)
            .expect("failed to open destination file");

        let start = 37_u64;
        let len = 131_072_u64;
        let progress = ProgressBar::hidden();
        copy_range_with_read_write(
            &source_file,
            &destination_file,
            start,
            len,
            64 * 1024,
            &progress,
        )
        .expect("range copy with read/write fallback should succeed");

        let copied = fs::read(&destination).expect("failed to read copied file");
        assert_eq!(
            &copied[start as usize..(start + len) as usize],
            &data[start as usize..(start + len) as usize]
        );
        assert!(copied[..start as usize].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn copy_range_copies_full_file() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.bin");
        let data = patterned_data(128 * 1024);
        write_file_with_mode(&source, &data, 0o644).expect("failed to write source file");

        let destination = temp.path().join("destination.bin");
        prepare_destination_file(&destination, data.len() as u64)
            .expect("failed to prepare destination file");

        let progress = ProgressBar::hidden();
        copy_range(
            &source,
            &destination,
            0,
            data.len() as u64,
            64 * 1024,
            &progress,
        )
        .expect("copy_range should copy full file");

        assert_eq!(
            fs::read(&destination).expect("failed to read destination file"),
            data
        );
    }

    #[test]
    fn copy_single_file_with_ranges_copies_file_contents() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("large-source.bin");
        let data = patterned_data(MIN_SINGLE_FILE_RANGE_SIZE + 2048);
        write_file_with_mode(&source, &data, 0o600).expect("failed to write large source file");

        let destination = temp.path().join("large-destination.bin");
        let job = file_job_from_paths(&source, &destination);

        let progress = ProgressBar::hidden();
        copy_single_file_with_ranges(&job, 4, MIN_CHUNK_SIZE, &progress)
            .expect("single-file range copy should succeed");

        assert_eq!(
            fs::read(&destination).expect("failed to read destination"),
            data
        );
    }

    #[test]
    fn copy_jobs_with_pool_copies_files_and_symlinks() {
        let temp = tempdir().expect("failed to create tempdir");
        let source_root = temp.path().join("source");
        fs::create_dir_all(source_root.join("nested")).expect("failed to create source tree");

        let file_one = source_root.join("one.txt");
        let file_two = source_root.join("nested/two.txt");
        write_file_with_mode(&file_one, b"one", 0o640).expect("failed to write file one");
        write_file_with_mode(&file_two, b"two", 0o600).expect("failed to write file two");

        let link = source_root.join("one-link");
        symlink(Path::new("one.txt"), &link).expect("failed to create symlink");

        let destination_root = temp.path().join("destination");
        let file_jobs = vec![
            file_job_from_paths(&file_one, &destination_root.join("one.txt")),
            file_job_from_paths(&file_two, &destination_root.join("nested/two.txt")),
        ];
        let symlink_jobs = vec![SymlinkJob {
            destination: destination_root.join("one-link"),
            target: fs::read_link(&link).expect("failed to read symlink target"),
            metadata: preserved_metadata_from(
                &fs::symlink_metadata(&link).expect("failed to read symlink metadata"),
            ),
        }];
        let plan = CopyPlan {
            file_jobs: file_jobs.clone(),
            symlink_jobs: symlink_jobs.clone(),
            directory_metadata: vec![
                (
                    destination_root.clone(),
                    PreservedMetadata {
                        mode: 0o755,
                        uid: 0,
                        gid: 0,
                        atime_sec: 0,
                        atime_nsec: 0,
                        mtime_sec: 0,
                        mtime_nsec: 0,
                    },
                ),
                (
                    destination_root.join("nested"),
                    PreservedMetadata {
                        mode: 0o755,
                        uid: 0,
                        gid: 0,
                        atime_sec: 0,
                        atime_nsec: 0,
                        mtime_sec: 0,
                        mtime_nsec: 0,
                    },
                ),
            ],
        };

        prepare_destination_layout(&plan).expect("failed to prepare destination layout");
        let progress = ProgressBar::hidden();
        copy_jobs_with_pool(&file_jobs, &symlink_jobs, 4, MIN_CHUNK_SIZE, &progress)
            .expect("copy pool should complete successfully");
        apply_preserved_metadata(&plan).expect("metadata application should succeed");

        assert_eq!(
            fs::read(destination_root.join("one.txt")).expect("failed to read copied file one"),
            b"one"
        );
        assert_eq!(
            fs::read(destination_root.join("nested/two.txt"))
                .expect("failed to read copied file two"),
            b"two"
        );
        let copied_symlink = destination_root.join("one-link");
        assert!(
            fs::symlink_metadata(&copied_symlink)
                .expect("failed to read copied symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_link(&copied_symlink).expect("failed to read copied symlink target"),
            PathBuf::from("one.txt")
        );
    }

    #[test]
    fn copy_jobs_with_pool_propagates_worker_errors() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.txt");
        write_file_with_mode(&source, b"source", 0o644).expect("failed to write source file");

        let destination_directory = temp.path().join("destination-dir");
        fs::create_dir(&destination_directory).expect("failed to create destination directory");

        let job = file_job_from_paths(&source, &destination_directory);
        let progress = ProgressBar::hidden();
        let error = copy_jobs_with_pool(&[job], &[], 1, MIN_CHUNK_SIZE, &progress)
            .expect_err("expected worker error for directory destination");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn apply_preserved_metadata_sets_file_and_directory_modes() {
        let temp = tempdir().expect("failed to create tempdir");
        let directory = temp.path().join("directory");
        fs::create_dir(&directory).expect("failed to create directory");

        let file = directory.join("file.txt");
        write_file_with_mode(&file, b"file", 0o644).expect("failed to write file");

        let plan = CopyPlan {
            file_jobs: vec![FileJob {
                source: file.clone(),
                destination: file.clone(),
                size: 4,
                metadata: PreservedMetadata {
                    mode: 0o600,
                    uid: fs::metadata(&file).expect("failed to stat file").uid(),
                    gid: fs::metadata(&file).expect("failed to stat file").gid(),
                    atime_sec: 1,
                    atime_nsec: 0,
                    mtime_sec: 2,
                    mtime_nsec: 0,
                },
            }],
            symlink_jobs: Vec::new(),
            directory_metadata: vec![(
                directory.clone(),
                PreservedMetadata {
                    mode: 0o711,
                    uid: fs::metadata(&directory)
                        .expect("failed to stat directory")
                        .uid(),
                    gid: fs::metadata(&directory)
                        .expect("failed to stat directory")
                        .gid(),
                    atime_sec: 3,
                    atime_nsec: 0,
                    mtime_sec: 4,
                    mtime_nsec: 0,
                },
            )],
        };

        apply_preserved_metadata(&plan).expect("metadata application should succeed");

        let file_mode = fs::symlink_metadata(&file)
            .expect("failed to read file metadata")
            .permissions()
            .mode()
            & 0o777;
        let dir_mode = fs::symlink_metadata(&directory)
            .expect("failed to read directory metadata")
            .permissions()
            .mode()
            & 0o777;
        let file_metadata = fs::symlink_metadata(&file).expect("failed to read file metadata");
        let dir_metadata =
            fs::symlink_metadata(&directory).expect("failed to read directory metadata");
        assert_eq!(file_mode, 0o600);
        assert_eq!(dir_mode, 0o711);
        assert_eq!(file_metadata.atime(), 1);
        assert_eq!(file_metadata.mtime(), 2);
        assert_eq!(dir_metadata.atime(), 3);
        assert_eq!(dir_metadata.mtime(), 4);
    }

    #[test]
    fn run_copies_directory_tree_end_to_end() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source");
        fs::create_dir_all(source.join("nested")).expect("failed to create source tree");

        write_file_with_mode(&source.join("root.txt"), b"root", 0o640)
            .expect("failed to write root file");
        let root_path = source.join("root.txt");
        let root_metadata = fs::symlink_metadata(&root_path).expect("failed to read root metadata");
        set_path_timestamps(
            &root_path,
            PreservedMetadata {
                mode: root_metadata.permissions().mode(),
                uid: root_metadata.uid(),
                gid: root_metadata.gid(),
                atime_sec: 1_234_567_890,
                atime_nsec: 123,
                mtime_sec: 1_234_567_891,
                mtime_nsec: 456,
            },
            true,
        )
        .expect("failed to set root timestamps");
        write_file_with_mode(&source.join(".hidden"), b"hidden", 0o600)
            .expect("failed to write hidden file");
        let nested_data = patterned_data(128 * 1024);
        write_file_with_mode(&source.join("nested/data.bin"), &nested_data, 0o600)
            .expect("failed to write nested data file");

        symlink(Path::new("nested/data.bin"), source.join("data-link"))
            .expect("failed to create data symlink");
        symlink(Path::new("missing-target"), source.join("broken-link"))
            .expect("failed to create broken symlink");
        let expected_root_metadata =
            fs::symlink_metadata(&root_path).expect("failed to read expected root metadata");

        let output_parent = temp.path().join("output-parent");
        fs::create_dir(&output_parent).expect("failed to create output parent directory");

        run(Args {
            paths: vec![source.clone(), output_parent.clone()],
            silent: true,
        })
        .expect("end-to-end copy should succeed");

        let copied_root = output_parent.join("source");
        let copied_metadata = fs::symlink_metadata(copied_root.join("root.txt"))
            .expect("failed to read copied metadata");
        assert_eq!(copied_metadata.atime(), expected_root_metadata.atime());
        assert_eq!(
            copied_metadata.atime_nsec(),
            expected_root_metadata.atime_nsec()
        );
        assert_eq!(copied_metadata.mtime(), expected_root_metadata.mtime());
        assert_eq!(
            copied_metadata.mtime_nsec(),
            expected_root_metadata.mtime_nsec()
        );

        assert_eq!(
            fs::read(copied_root.join("root.txt")).expect("failed to read copied root file"),
            b"root"
        );
        assert_eq!(
            fs::read(copied_root.join(".hidden")).expect("failed to read copied hidden file"),
            b"hidden"
        );
        assert_eq!(
            fs::read(copied_root.join("nested/data.bin"))
                .expect("failed to read copied nested file"),
            nested_data
        );

        let copied_link = copied_root.join("data-link");
        assert!(
            fs::symlink_metadata(&copied_link)
                .expect("failed to read copied symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_link(&copied_link).expect("failed to read copied symlink target"),
            PathBuf::from("nested/data.bin")
        );

        let copied_broken = copied_root.join("broken-link");
        assert!(
            fs::symlink_metadata(&copied_broken)
                .expect("failed to read copied broken symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_link(&copied_broken).expect("failed to read copied broken symlink target"),
            PathBuf::from("missing-target")
        );

        let copied_mode = fs::symlink_metadata(copied_root.join("root.txt"))
            .expect("failed to read copied mode")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(copied_mode, 0o640);
    }

    #[test]
    fn run_copies_top_level_symlink() {
        let temp = tempdir().expect("failed to create tempdir");
        let source_dir = temp.path().join("source");
        fs::create_dir(&source_dir).expect("failed to create source directory");
        write_file_with_mode(&source_dir.join("target.txt"), b"target", 0o644)
            .expect("failed to write target file");

        let top_level_symlink = temp.path().join("top-level-link");
        symlink(Path::new("source/target.txt"), &top_level_symlink)
            .expect("failed to create top-level symlink");

        let output_dir = temp.path().join("output");
        fs::create_dir(&output_dir).expect("failed to create output directory");

        run(Args {
            paths: vec![top_level_symlink.clone(), output_dir.clone()],
            silent: true,
        })
        .expect("top-level symlink copy should succeed");

        let copied_symlink = output_dir.join("top-level-link");
        assert!(
            fs::symlink_metadata(&copied_symlink)
                .expect("failed to read copied symlink metadata")
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            fs::read_link(&copied_symlink).expect("failed to read copied symlink target"),
            PathBuf::from("source/target.txt")
        );
    }

    #[test]
    fn run_rejects_copying_directory_into_itself() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source");
        fs::create_dir(&source).expect("failed to create source directory");
        write_file_with_mode(&source.join("file.txt"), b"file", 0o644)
            .expect("failed to write source file");

        let error = run(Args {
            paths: vec![source.clone(), source.join("nested-output")],
            silent: true,
        })
        .expect_err("copying into itself should fail");
        assert!(
            error
                .to_string()
                .contains("cannot copy a directory into itself")
        );
    }

    #[test]
    fn resolve_destination_metadata_error_paths_are_reported() {
        let temp = tempdir().expect("failed to create tempdir");
        let source_file = temp.path().join("source.txt");
        write_file_with_mode(&source_file, b"source", 0o644).expect("failed to write source");
        let source_dir = temp.path().join("source-dir");
        fs::create_dir(&source_dir).expect("failed to create source directory");

        let blocking_file = temp.path().join("blocking-file");
        write_file_with_mode(&blocking_file, b"block", 0o644)
            .expect("failed to write blocking file");
        let impossible_output = blocking_file.join("child");

        let file_error = resolve_file_destination(&source_file, &impossible_output)
            .expect_err("resolve_file_destination should return metadata errors");
        assert_ne!(file_error.kind(), io::ErrorKind::NotFound);

        let dir_error = resolve_directory_destination(&source_dir, &impossible_output)
            .expect_err("resolve_directory_destination should return metadata errors");
        assert_ne!(dir_error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn resolve_directory_destination_requires_directory_name() {
        let temp = tempdir().expect("failed to create tempdir");
        let output = temp.path().join("output");
        fs::create_dir(&output).expect("failed to create output directory");

        let error = resolve_directory_destination(Path::new("/"), &output)
            .expect_err("root input should be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn is_path_within_base_handles_mixed_path_styles() {
        assert!(!is_path_within_base(
            Path::new("relative/path"),
            Path::new("/absolute/base")
        ));
    }

    #[test]
    fn ensure_destination_outside_source_handles_existing_and_relative_outputs() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source");
        fs::create_dir(&source).expect("failed to create source directory");

        let existing_destination = temp.path().join("existing-destination");
        fs::create_dir(&existing_destination).expect("failed to create existing destination");
        ensure_destination_outside_source(&source, &existing_destination)
            .expect("existing destination outside source should be allowed");

        ensure_destination_outside_source(&source, Path::new("relative-destination-path"))
            .expect("relative destination outside source should be allowed");
    }

    #[test]
    fn build_copy_plan_for_regular_file_succeeds() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.txt");
        write_file_with_mode(&source, b"source-data", 0o640).expect("failed to write source");

        let destination = temp.path().join("destination.txt");
        let plan =
            build_copy_plan(&source, &destination, 2).expect("file copy plan should be created");

        assert_eq!(plan.file_jobs.len(), 1);
        assert_eq!(plan.symlink_jobs.len(), 0);
        assert_eq!(plan.directory_metadata.len(), 0);
        assert_eq!(plan.file_jobs[0].destination, destination);
        assert_eq!(plan.file_jobs[0].metadata.mode & 0o777, 0o640);
    }

    #[test]
    fn build_copy_plan_for_sources_requires_existing_directory_with_multiple_inputs() {
        let temp = tempdir().expect("failed to create tempdir");
        let source_one = temp.path().join("source-one.txt");
        let source_two = temp.path().join("source-two.txt");
        write_file_with_mode(&source_one, b"one", 0o640).expect("failed to write source one");
        write_file_with_mode(&source_two, b"two", 0o640).expect("failed to write source two");

        let destination_file = temp.path().join("destination.txt");
        write_file_with_mode(&destination_file, b"destination", 0o640)
            .expect("failed to write destination file");

        let error = build_copy_plan_for_sources(&[source_one, source_two], &destination_file, 2)
            .expect_err("multiple sources should require destination directory");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn build_copy_plan_rejects_same_top_level_symlink() {
        let temp = tempdir().expect("failed to create tempdir");
        let target = temp.path().join("target.txt");
        write_file_with_mode(&target, b"target", 0o644).expect("failed to write target");
        let link = temp.path().join("link.txt");
        symlink(Path::new("target.txt"), &link).expect("failed to create symlink");

        let error =
            build_copy_plan(&link, &link, 2).expect_err("same symlink source/destination fails");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn build_copy_plan_rejects_top_level_fifo() {
        let temp = tempdir().expect("failed to create tempdir");
        let fifo = temp.path().join("top-level-fifo");
        let fifo_c_string =
            CString::new(fifo.as_os_str().as_bytes()).expect("failed to build fifo CString");
        let mkfifo_result = unsafe { libc::mkfifo(fifo_c_string.as_ptr(), 0o644) };
        assert_eq!(mkfifo_result, 0, "mkfifo should succeed");

        let destination = temp.path().join("destination");
        let error =
            build_copy_plan(&fifo, &destination, 2).expect_err("top-level fifo should be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn remove_existing_non_directory_destination_propagates_metadata_errors() {
        let temp = tempdir().expect("failed to create tempdir");
        let blocking_file = temp.path().join("blocking-file");
        write_file_with_mode(&blocking_file, b"block", 0o644)
            .expect("failed to write blocking file");

        let impossible_path = blocking_file.join("child");
        let error = remove_existing_non_directory_destination(&impossible_path)
            .expect_err("metadata error should be propagated");
        assert_ne!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn copy_range_with_read_write_reports_unexpected_eof_and_io_errors() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.bin");
        write_file_with_mode(&source, b"abcd", 0o644).expect("failed to write source");

        let destination = temp.path().join("destination.bin");
        prepare_destination_file(&destination, 16).expect("failed to prepare destination");
        let source_file = OpenOptions::new()
            .read(true)
            .open(&source)
            .expect("failed to open source file");
        let destination_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&destination)
            .expect("failed to open destination file");

        let progress = ProgressBar::hidden();
        let eof_error = copy_range_with_read_write(
            &source_file,
            &destination_file,
            0,
            16,
            64 * 1024,
            &progress,
        )
        .expect_err("copy beyond EOF should fail");
        assert_eq!(eof_error.kind(), io::ErrorKind::UnexpectedEof);

        let source_write_only = OpenOptions::new()
            .write(true)
            .open(&source)
            .expect("failed to open source in write-only mode");
        let read_error = copy_range_with_read_write(
            &source_write_only,
            &destination_file,
            0,
            1,
            64 * 1024,
            &progress,
        )
        .expect_err("write-only source should fail read_at");
        assert_eq!(read_error.raw_os_error(), Some(libc::EBADF));

        let destination_read_only = OpenOptions::new()
            .read(true)
            .open(&destination)
            .expect("failed to open destination in read-only mode");
        let write_error = copy_range_with_read_write(
            &source_file,
            &destination_read_only,
            0,
            1,
            64 * 1024,
            &progress,
        )
        .expect_err("read-only destination should fail write_at");
        assert_eq!(write_error.raw_os_error(), Some(libc::EBADF));
    }

    #[test]
    fn copy_range_with_syscall_covers_error_paths() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.bin");
        write_file_with_mode(&source, b"abcdef", 0o644).expect("failed to write source");
        let destination = temp.path().join("destination.bin");
        prepare_destination_file(&destination, 6).expect("failed to prepare destination");

        let progress = ProgressBar::hidden();
        let mut interrupted_then_error = 0;
        let interrupted_error = copy_range_with_syscall(
            &source,
            &destination,
            0,
            6,
            64 * 1024,
            &progress,
            |_, _, _, _, _| {
                interrupted_then_error += 1;
                unsafe {
                    let errno = libc::__errno_location();
                    *errno = if interrupted_then_error == 1 {
                        libc::EINTR
                    } else {
                        libc::EACCES
                    };
                }
                -1
            },
        )
        .expect_err("non-fallback syscall error should propagate");
        assert_eq!(interrupted_error.raw_os_error(), Some(libc::EACCES));

        let fallback_progress = ProgressBar::hidden();
        copy_range_with_syscall(
            &source,
            &destination,
            0,
            6,
            64 * 1024,
            &fallback_progress,
            |_, _, _, _, _| {
                unsafe {
                    *libc::__errno_location() = libc::EINVAL;
                }
                -1
            },
        )
        .expect("fallback path should copy with read/write");
        assert_eq!(
            fs::read(&destination).expect("failed to read destination"),
            b"abcdef"
        );

        let eof_progress = ProgressBar::hidden();
        let eof_error = copy_range_with_syscall(
            &source,
            &destination,
            0,
            1,
            64 * 1024,
            &eof_progress,
            |_, _, _, _, _| 0,
        )
        .expect_err("zero-byte syscall result should fail");
        assert_eq!(eof_error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn copy_range_rejects_large_start_offsets() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.bin");
        write_file_with_mode(&source, b"source", 0o644).expect("failed to write source");
        let destination = temp.path().join("destination.bin");
        prepare_destination_file(&destination, 6).expect("failed to prepare destination");

        let progress = ProgressBar::hidden();
        let error = copy_range(
            &source,
            &destination,
            (i64::MAX as u64) + 1,
            1,
            64 * 1024,
            &progress,
        )
        .expect_err("offsets larger than i64 should be rejected");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn copy_file_job_and_single_file_ranges_handle_empty_files() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("empty-source.bin");
        write_file_with_mode(&source, b"", 0o644).expect("failed to write empty source");

        let destination_job = temp.path().join("empty-job.bin");
        let job = file_job_from_paths(&source, &destination_job);
        copy_file_job(&job, MIN_CHUNK_SIZE, &ProgressBar::hidden())
            .expect("empty file copy job should succeed");
        assert_eq!(
            fs::metadata(&destination_job)
                .expect("failed to stat destination")
                .len(),
            0
        );

        let destination_ranges = temp.path().join("empty-ranges.bin");
        let ranges_job = file_job_from_paths(&source, &destination_ranges);
        copy_single_file_with_ranges(&ranges_job, 4, MIN_CHUNK_SIZE, &ProgressBar::hidden())
            .expect("empty single-file range copy should succeed");
        assert_eq!(
            fs::metadata(&destination_ranges)
                .expect("failed to stat range destination")
                .len(),
            0
        );
    }

    #[test]
    fn join_worker_helpers_cover_success_error_and_panic_paths() {
        let ok_join = thread::spawn(|| Ok::<(), io::Error>(()));
        join_io_worker(ok_join).expect("successful io worker should join");

        let err_join = thread::spawn(|| Err::<(), io::Error>(io::Error::other("worker error")));
        let worker_error = join_io_worker(err_join).expect_err("io worker error should propagate");
        assert_eq!(worker_error.kind(), io::ErrorKind::Other);

        let panic_join = thread::spawn(|| -> io::Result<()> { panic!("worker panic") });
        let panic_error = join_io_worker(panic_join).expect_err("panic should become io error");
        assert_eq!(panic_error.kind(), io::ErrorKind::Other);

        let ok_unit_join = thread::spawn(|| {});
        join_unit_worker(ok_unit_join).expect("successful unit worker should join");

        let panic_unit_join = thread::spawn(|| panic!("unit worker panic"));
        let panic_unit_error =
            join_unit_worker(panic_unit_join).expect_err("unit panic should become io error");
        assert_eq!(panic_unit_error.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn copy_jobs_with_pool_accepts_empty_job_list() {
        copy_jobs_with_pool(&[], &[], 4, MIN_CHUNK_SIZE, &ProgressBar::hidden())
            .expect("empty job list should be a no-op");
    }

    #[test]
    fn run_copies_single_file_and_surfaces_copy_errors() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source-file.txt");
        write_file_with_mode(&source, b"file-content", 0o644).expect("failed to write source");

        let destination = temp.path().join("destination-file.txt");
        run(Args {
            paths: vec![source.clone(), destination.clone()],
            silent: true,
        })
        .expect("single file run should succeed");
        assert_eq!(
            fs::read(&destination).expect("failed to read destination"),
            b"file-content"
        );

        let output_dir = temp.path().join("output-directory");
        fs::create_dir(&output_dir).expect("failed to create output directory");
        fs::create_dir(output_dir.join("source-file.txt"))
            .expect("failed to create destination collision directory");

        let error = run(Args {
            paths: vec![source, output_dir],
            silent: true,
        })
        .expect_err("run should surface copy failures");
        assert!(error.to_string().contains("existing directory"));
    }

    #[test]
    fn collect_directory_jobs_reports_walk_errors() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("missing-source");

        let destination = temp.path().join("destination");
        let mut plan = CopyPlan::default();
        let result = collect_directory_jobs(&source, &destination, 2, &mut plan);

        assert!(result.is_err(), "walk should fail for missing source paths");
    }

    #[test]
    fn prepare_destination_helpers_handle_parentless_paths() {
        let temp = tempdir().expect("failed to create tempdir");
        let source = temp.path().join("source.txt");
        write_file_with_mode(&source, b"source", 0o644).expect("failed to write source");

        let plan = CopyPlan {
            file_jobs: vec![file_job_from_paths(&source, Path::new("/"))],
            symlink_jobs: vec![SymlinkJob {
                destination: PathBuf::from("/"),
                target: PathBuf::from("target"),
                metadata: preserved_metadata_from(
                    &fs::symlink_metadata(&source).expect("failed to read source metadata"),
                ),
            }],
            directory_metadata: Vec::new(),
        };
        prepare_destination_layout(&plan)
            .expect("parentless destinations should be skipped during layout prep");

        let error = prepare_destination_file(Path::new("/"), 0)
            .expect_err("opening root as a file should fail");
        assert_eq!(error.kind(), io::ErrorKind::IsADirectory);
    }

    #[test]
    fn copy_jobs_with_pool_sets_failure_flag_for_other_workers() {
        let temp = tempdir().expect("failed to create tempdir");
        let source_root = temp.path().join("source-root");
        fs::create_dir_all(&source_root).expect("failed to create source root");

        let failing_source = source_root.join("failing.txt");
        write_file_with_mode(&failing_source, b"failing", 0o644)
            .expect("failed to write failing source");
        let failing_destination = temp.path().join("destination-collision");
        fs::create_dir(&failing_destination).expect("failed to create destination collision");
        let failing_job = file_job_from_paths(&failing_source, &failing_destination);

        let valid_source = source_root.join("valid.bin");
        let valid_data = patterned_data(MIN_SINGLE_FILE_RANGE_SIZE + 8192);
        write_file_with_mode(&valid_source, &valid_data, 0o644)
            .expect("failed to write valid source");

        let valid_destination_root = temp.path().join("valid-destination-root");
        let mut file_jobs = vec![failing_job];
        for index in 0..6 {
            file_jobs.push(file_job_from_paths(
                &valid_source,
                &valid_destination_root.join(format!("copy-{index}.bin")),
            ));
        }

        let plan = CopyPlan {
            file_jobs: file_jobs.clone(),
            symlink_jobs: Vec::new(),
            directory_metadata: vec![(
                valid_destination_root,
                PreservedMetadata {
                    mode: 0o755,
                    uid: 0,
                    gid: 0,
                    atime_sec: 0,
                    atime_nsec: 0,
                    mtime_sec: 0,
                    mtime_nsec: 0,
                },
            )],
        };
        prepare_destination_layout(&plan).expect("failed to prepare destination layout");

        let error = copy_jobs_with_pool(&file_jobs, &[], 6, MIN_CHUNK_SIZE, &ProgressBar::hidden())
            .expect_err("expected failure from destination collision");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }
}
