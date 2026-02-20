# cpb

A small utility for faster copying of files and directories, especially for ceph and nfs based filesystems.

In my tests, it can be up to 10x faster than `cp`. The main reason for this is that it uses 1. multiple threads to copy files, it also uses the `copy_file_range` syscall on linux, which allows it to copy files without having to read them into user space and it uses way larger buffers than `cp` (32 MB vs. 128 kB).

This comes with a cost of higher memory usage. It also intentionally removes almost all flags that `cp` has, so it will "just" copy files and directories, without any of the fancy features that `cp` has. This is intentional, as I want to keep the code simple and focused on the main goal of copying files as fast as possible.


## Install 

You can install cpb using cargo:

```bash
cargo install cpb
```

## Usage

```bash
cpb [SOURCE(s)] [DESTINATION]
```

That's it. You can specify multiple sources, but the destination must be a directory. If you specify only one source, the destination can be a file or a directory. No flags, no options, just copy files and directories. There is a single flag `--silent` that will suppress all output, but other than that, there are no options.

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.