use std::fs;
use std::process::Command;

use tempfile::tempdir;

#[test]
fn cli_main_copies_a_file() {
    let temp = tempdir().expect("failed to create tempdir");
    let source = temp.path().join("source.txt");
    fs::write(&source, b"copied via cli").expect("failed to write source");

    let destination = temp.path().join("destination.txt");
    let status = Command::new(env!("CARGO_BIN_EXE_cpb"))
        .arg(&source)
        .arg(&destination)
        .arg("--threads")
        .arg("2")
        .arg("--chunk-size")
        .arg((32 * 1024 * 1024).to_string())
        .status()
        .expect("failed to execute cpb");

    assert!(status.success(), "cpb exited with non-zero status");
    assert_eq!(
        fs::read(&destination).expect("failed to read copied destination"),
        b"copied via cli"
    );
}
