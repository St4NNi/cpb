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
        .status()
        .expect("failed to execute cpb");

    assert!(status.success(), "cpb exited with non-zero status");
    assert_eq!(
        fs::read(&destination).expect("failed to read copied destination"),
        b"copied via cli"
    );
}

#[test]
fn cli_main_copies_multiple_sources_into_directory() {
    let temp = tempdir().expect("failed to create tempdir");
    let source_one = temp.path().join("source-one.txt");
    let source_two = temp.path().join("source-two.txt");
    fs::write(&source_one, b"source one").expect("failed to write source one");
    fs::write(&source_two, b"source two").expect("failed to write source two");

    let destination = temp.path().join("destination-dir");
    fs::create_dir(&destination).expect("failed to create destination dir");

    let status = Command::new(env!("CARGO_BIN_EXE_cpb"))
        .arg(&source_one)
        .arg(&source_two)
        .arg(&destination)
        .status()
        .expect("failed to execute cpb");

    assert!(status.success(), "cpb exited with non-zero status");
    assert_eq!(
        fs::read(destination.join("source-one.txt")).expect("failed to read copied source one"),
        b"source one"
    );
    assert_eq!(
        fs::read(destination.join("source-two.txt")).expect("failed to read copied source two"),
        b"source two"
    );
}
