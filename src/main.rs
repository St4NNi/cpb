use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::VecDeque;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
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

#[derive(Debug, Parser)]
#[command(name = "bettercp", about = "Parallel copy with copy_file_range")]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[arg(
        short,
        long,
        default_value_t = 32,
        value_parser = parse_threads
    )]
    threads: usize,
    #[arg(
        short = 'c',
        long,
        default_value_t = MIN_CHUNK_SIZE,
        value_parser = parse_chunk_size
    )]
    chunk_size: usize,
}

#[derive(Debug, Clone)]
struct FileJob {
    source: PathBuf,
    destination: PathBuf,
    size: u64,
    mode: u32,
}

#[derive(Debug, Default)]
struct CopyPlan {
    jobs: Vec<FileJob>,
    directory_modes: Vec<(PathBuf, u32)>,
}

fn parse_threads(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "threads must be an integer".to_owned())?;
    if (1..=32).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("threads must be in the range 1..=32".to_owned())
    }
}

fn parse_chunk_size(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "chunk-size must be an integer".to_owned())?;
    if parsed >= MIN_CHUNK_SIZE {
        Ok(parsed)
    } else {
        Err(format!(
            "chunk-size must be at least {MIN_CHUNK_SIZE} bytes (32 MiB)"
        ))
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

fn ensure_destination_outside_source(source: &Path, destination_root: &Path) -> io::Result<()> {
    let source_canonical = fs::canonicalize(source)?;

    let destination_approx = if destination_root.exists() {
        fs::canonicalize(destination_root)?
    } else if destination_root.is_absolute() {
        destination_root.to_path_buf()
    } else {
        env::current_dir()?.join(destination_root)
    };

    if destination_approx.starts_with(&source_canonical) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot copy a directory into itself",
        ));
    }

    Ok(())
}

fn collect_directory_jobs(
    source_directory: &Path,
    destination_directory: &Path,
    plan: &mut CopyPlan,
) -> io::Result<()> {
    let metadata = fs::symlink_metadata(source_directory)?;
    plan.directory_modes.push((
        destination_directory.to_path_buf(),
        metadata.permissions().mode(),
    ));

    for entry_result in fs::read_dir(source_directory)? {
        let entry = entry_result?;
        let source_path = entry.path();
        let destination_path = destination_directory.join(entry.file_name());
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            collect_directory_jobs(&source_path, &destination_path, plan)?;
            continue;
        }

        if file_type.is_file() {
            let file_metadata = entry.metadata()?;
            plan.jobs.push(FileJob {
                source: source_path,
                destination: destination_path,
                size: file_metadata.len(),
                mode: file_metadata.permissions().mode(),
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

fn build_copy_plan(input: &Path, output: &Path) -> io::Result<CopyPlan> {
    let source_metadata = fs::symlink_metadata(input)?;

    if source_metadata.is_file() {
        let destination = resolve_file_destination(input, output)?;

        if let Ok(destination_metadata) = fs::metadata(&destination) {
            if source_metadata.dev() == destination_metadata.dev()
                && source_metadata.ino() == destination_metadata.ino()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "input and output point to the same file",
                ));
            }
        }

        return Ok(CopyPlan {
            jobs: vec![FileJob {
                source: input.to_path_buf(),
                destination,
                size: source_metadata.len(),
                mode: source_metadata.permissions().mode(),
            }],
            directory_modes: Vec::new(),
        });
    }

    if source_metadata.is_dir() {
        let destination_root = resolve_directory_destination(input, output)?;
        ensure_destination_outside_source(input, &destination_root)?;

        let mut plan = CopyPlan::default();
        collect_directory_jobs(input, &destination_root, &mut plan)?;
        return Ok(plan);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "input must be a regular file or directory",
    ))
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
    for (directory, _) in &plan.directory_modes {
        fs::create_dir_all(directory)?;
    }

    for job in &plan.jobs {
        if let Some(parent) = job.destination.parent() {
            fs::create_dir_all(parent)?;
        }

        let output = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&job.destination)?;

        output.set_len(job.size)?;
    }

    Ok(())
}

fn should_fallback_copy_file_range(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EXDEV | libc::EOPNOTSUPP | libc::ENOSYS | libc::EINVAL)
    )
}

fn copy_range_with_read_write(
    source: &File,
    destination: &File,
    start: u64,
    len: u64,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    let io_chunk_size = chunk_size.min(FALLBACK_IO_CHUNK_SIZE).max(64 * 1024);
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

fn copy_range(
    input: &Path,
    output: &Path,
    start: u64,
    len: u64,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
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
    let mut out_offset = i64::try_from(start).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "destination offset exceeds supported range",
        )
    })?;

    let mut remaining = len;
    while remaining > 0 {
        let to_copy = remaining.min(chunk_size as u64) as usize;
        let copied = unsafe {
            libc::copy_file_range(
                src_fd,
                &mut in_offset as *mut libc::loff_t,
                dst_fd,
                &mut out_offset as *mut libc::loff_t,
                to_copy,
                0,
            )
        };

        if copied < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }

            if should_fallback_copy_file_range(&error) {
                let offset = u64::try_from(in_offset).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "copy offset exceeds supported range",
                    )
                })?;

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

fn copy_single_file_with_ranges(
    job: &FileJob,
    threads: usize,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    if job.size == 0 {
        return Ok(());
    }

    let worker_count = single_file_worker_count(job.size, threads);
    let ranges = split_ranges(job.size, worker_count);

    let mut handles = Vec::with_capacity(worker_count);
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
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(io::Error::other("copy worker panicked")),
        }
    }

    Ok(())
}

fn copy_jobs_with_pool(
    jobs: &[FileJob],
    threads: usize,
    chunk_size: usize,
    progress: &ProgressBar,
) -> io::Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }

    let worker_count = threads.min(jobs.len()).max(1);
    let queue = Arc::new(Mutex::new((0..jobs.len()).collect::<VecDeque<_>>()));
    let shared_jobs = Arc::new(jobs.to_vec());
    let has_failed = Arc::new(AtomicBool::new(false));
    let first_error = Arc::new(Mutex::new(None::<io::Error>));

    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let queue = Arc::clone(&queue);
        let shared_jobs = Arc::clone(&shared_jobs);
        let has_failed = Arc::clone(&has_failed);
        let first_error = Arc::clone(&first_error);
        let thread_progress = progress.clone();

        handles.push(thread::spawn(move || {
            loop {
                if has_failed.load(Ordering::Relaxed) {
                    break;
                }

                let next_job = {
                    let mut queue = queue.lock().expect("copy queue mutex poisoned");
                    queue.pop_front()
                };

                let Some(index) = next_job else {
                    break;
                };

                let job = &shared_jobs[index];
                if let Err(error) = copy_range(
                    &job.source,
                    &job.destination,
                    0,
                    job.size,
                    chunk_size,
                    &thread_progress,
                ) {
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
        if handle.join().is_err() {
            return Err(io::Error::other("copy worker panicked"));
        }
    }

    let mut slot = first_error
        .lock()
        .map_err(|_| io::Error::other("error slot mutex poisoned"))?;
    if let Some(error) = slot.take() {
        return Err(error);
    }

    Ok(())
}

fn apply_permissions(plan: &CopyPlan) -> io::Result<()> {
    for job in &plan.jobs {
        fs::set_permissions(&job.destination, fs::Permissions::from_mode(job.mode))?;
    }

    for (directory, mode) in plan.directory_modes.iter().rev() {
        fs::set_permissions(directory, fs::Permissions::from_mode(*mode))?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let plan = build_copy_plan(&args.input, &args.output)?;

    let total_size = plan.jobs.iter().map(|job| job.size).sum::<u64>();
    let total_files = plan.jobs.len();

    let progress = ProgressBar::new(total_size);
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta}) {msg}",
        )?
        .progress_chars("=> "),
    );
    progress.enable_steady_tick(Duration::from_millis(120));
    progress.set_message("preparing destination");
    progress.tick();

    prepare_destination_layout(&plan)?;

    progress.set_message("copying");

    let copy_result = if total_files == 1 {
        copy_single_file_with_ranges(&plan.jobs[0], args.threads, args.chunk_size, &progress)
    } else {
        copy_jobs_with_pool(&plan.jobs, args.threads, args.chunk_size, &progress)
    };

    if let Err(error) = copy_result {
        progress.abandon();
        return Err(error.into());
    }

    apply_permissions(&plan)?;
    progress.finish_with_message(format!(
        "copied {total_size} bytes across {total_files} file(s)"
    ));

    Ok(())
}
