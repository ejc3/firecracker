// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provides serialization and deserialization facilities and implements a persistent storage
//! format for Firecracker state snapshots.
//!
//! The `Snapshot` API manages serialization and deserialization of collections of objects
//! that implement the `serde` `Serialize`, `Deserialize` trait. Currently, we use
//! [`bitcode`](https://docs.rs/bitcode/latest/bitcode/) for performing the serialization.
//!
//! The snapshot format uses the following layout:
//!
//!  |-----------------------------|
//!  |       64 bit magic_id       |
//!  |-----------------------------|
//!  |       version string        |
//!  |-----------------------------|
//!  |            State            |
//!  |-----------------------------|
//!  |        optional CRC64       |
//!  |-----------------------------|
//!
//!
//! The snapshot format uses a version value in the form of `MAJOR.MINOR.PATCH`. The version is
//! provided by the library clients (it is not tied to this crate).
pub mod crc;
mod persist;
use std::fmt::Debug;
use std::io::{Read, Write};

use crc64::crc64;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::persist::{FCVM_SNAPSHOT_LAYOUT, SNAPSHOT_VERSION};
use crate::snapshot::crc::CRC64Writer;
pub use crate::snapshot::persist::Persist;
use crate::utils::Version;

#[cfg(target_arch = "x86_64")]
const SNAPSHOT_MAGIC_ID: u64 = 0x0710_1984_8664_0000u64;

#[cfg(target_arch = "aarch64")]
const SNAPSHOT_MAGIC_ID: u64 = 0x0710_1984_AAAA_0000u64;

/// Maximum size in bytes for snapshot deserialization to prevent DOS attacks.
/// Snapshots contain VM state which can be large, but we set a reasonable upper bound.
/// This limit is 10MB which should be sufficient for any legitimate snapshot.
const SNAPSHOT_DESERIALIZATION_BYTES_LIMIT: usize = 10_000_000;

/// The first eight bytes of the layout mark that ends a snapshot file: "FCVMSNAP".
const LAYOUT_MARK_MAGIC: [u8; 8] = *b"FCVMSNAP";
/// Length of the layout mark: [`LAYOUT_MARK_MAGIC`], the layout number, and the layout number
/// with every bit inverted, both little-endian.
const LAYOUT_MARK_LEN: usize = 16;

/// The layout mark of a snapshot whose state has layout `layout`.
fn layout_mark(layout: u32) -> [u8; LAYOUT_MARK_LEN] {
    let mut mark = [0; LAYOUT_MARK_LEN];
    mark[..8].copy_from_slice(&LAYOUT_MARK_MAGIC);
    mark[8..12].copy_from_slice(&layout.to_le_bytes());
    mark[12..].copy_from_slice(&(!layout).to_le_bytes());
    mark
}

/// Checks that `buf`, the contents of a snapshot file, ends with the mark of the layout this
/// build reads ([`FCVM_SNAPSHOT_LAYOUT`]) and returns what precedes it: the serialized
/// snapshot and its CRC.
///
/// The mark is checked before anything is decoded, and it is the last thing in the file, after
/// the CRC. A file without one was written by a build that did not write marks, whatever
/// version it holds, and a build that does not know the mark takes the mark's last eight
/// bytes for the CRC, which then cannot match.
fn strip_layout_mark(buf: &[u8]) -> Result<&[u8], SnapshotError> {
    let Some((snapshot, mark)) = buf.split_last_chunk::<LAYOUT_MARK_LEN>() else {
        return Err(SnapshotError::NoLayoutMark);
    };
    let layout = u32::from_le_bytes(mark[8..12].try_into().unwrap());
    if *mark != layout_mark(layout) {
        return Err(SnapshotError::NoLayoutMark);
    }
    if layout != FCVM_SNAPSHOT_LAYOUT {
        return Err(SnapshotError::InvalidLayout(layout, FCVM_SNAPSHOT_LAYOUT));
    }
    Ok(snapshot)
}

/// Error definitions for the Snapshot API.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SnapshotError {
    /// CRC64 validation failed
    Crc64,
    /// Invalid data version: {0}
    InvalidFormatVersion(Version),
    /// Invalid data version: the snapshot has no layout mark, so another Firecracker build wrote it
    NoLayoutMark,
    /// Invalid data version: the snapshot's state has layout {0} and this build reads layout {1}
    InvalidLayout(u32, u32),
    /// Magic value does not match arch: {0}
    InvalidMagic(u64),
    /// An error occurred during bitcode serialization: {0}
    Bitcode(#[from] bitcode::Error),
    /// IO Error: {0}
    Io(#[from] std::io::Error),
    /// Snapshot size exceeds limit of {0} bytes
    SizeLimitExceeded(usize),
}

fn serialize<S: Serialize, W: Write>(data: &S, write: &mut W) -> Result<(), SnapshotError> {
    let encoded = bitcode::serialize(data)?;
    write.write_all(&encoded).map_err(SnapshotError::Io)
}

/// Firecracker snapshot header
#[derive(Debug, Serialize, Deserialize)]
struct SnapshotHdr {
    /// magic value
    magic: u64,
    /// Snapshot data version
    version: Version,
}

/// Assumes the raw bytes stream read from the given [`Read`] instance is a snapshot file,
/// and returns the version of it.
pub fn get_format_version<R: Read>(reader: &mut R) -> Result<Version, SnapshotError> {
    // Check size limit before reading the full file to prevent DOS attacks
    let mut buf = Vec::new();
    let bytes_read = reader
        .take((SNAPSHOT_DESERIALIZATION_BYTES_LIMIT + 1) as u64)
        .read_to_end(&mut buf)?;

    if bytes_read > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
        return Err(SnapshotError::SizeLimitExceeded(
            SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
        ));
    }

    format_version_from_buf(strip_layout_mark(&buf)?)
}

/// Returns the version of the snapshot in `buf`, the contents of a snapshot file including its
/// trailing CRC. Not generic, so it is compiled only once.
fn format_version_from_buf(buf: &[u8]) -> Result<Version, SnapshotError> {
    // The last 8 bytes are the CRC, so we need to separate them for deserialization
    if buf.len() < 8 {
        return Err(SnapshotError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "File too short to contain CRC",
        )));
    }

    let (data_buf, _crc_buf) = buf.split_at(buf.len() - 8);

    // Since bitcode requires exact type matching, we need to try deserializing
    // as the specific snapshot type we know about. In practice, all snapshots
    // in Firecracker use MicrovmState as the data type.
    use crate::persist::MicrovmState;

    match bitcode::deserialize::<Snapshot<MicrovmState>>(data_buf) {
        Ok(snapshot) => Ok(snapshot.header.version),
        Err(e) => {
            // If deserialization fails, it could be due to:
            // 1. The snapshot was created with bincode (older versions)
            // 2. The MicrovmState structure has changed and is incompatible
            // 3. The snapshot file is corrupted
            // Since supporting bincode is out of scope, we return a descriptive error.
            Err(SnapshotError::Bitcode(e))
        }
    }
}

/// Firecracker snapshot type
///
/// A type used to store and load Firecracker snapshots of a particular version
#[derive(Debug, Serialize, Deserialize)]
pub struct Snapshot<Data> {
    header: SnapshotHdr,
    /// The data stored int his [`Snapshot`]
    pub data: Data,
}

impl<Data> Snapshot<Data> {
    /// Constructs a new snapshot with the given `data`.
    pub fn new(data: Data) -> Self {
        Self {
            header: SnapshotHdr {
                magic: SNAPSHOT_MAGIC_ID,
                version: SNAPSHOT_VERSION,
            },
            data,
        }
    }

    /// Gets the version of this snapshot
    pub fn version(&self) -> &Version {
        &self.header.version
    }
}

impl<Data: DeserializeOwned> Snapshot<Data> {
    pub(crate) fn load_without_crc_check(buf: &[u8]) -> Result<Self, SnapshotError> {
        // Check size limit to prevent DOS attacks
        if buf.len() > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
            return Err(SnapshotError::SizeLimitExceeded(
                SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
            ));
        }

        let snapshot: Self = bitcode::deserialize(buf)?;

        // Validate the header
        if snapshot.header.magic != SNAPSHOT_MAGIC_ID {
            return Err(SnapshotError::InvalidMagic(snapshot.header.magic));
        }

        if snapshot.header.version.major() != SNAPSHOT_VERSION.major()
            || snapshot.header.version.minor() > SNAPSHOT_VERSION.minor()
        {
            return Err(SnapshotError::InvalidFormatVersion(snapshot.header.version));
        }

        Ok(snapshot)
    }

    /// Loads a snapshot from the given [`Read`] instance, performing all validations
    /// (CRC, snapshot magic value, snapshot version).
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, SnapshotError> {
        // Check size limit before reading the full file to prevent DOS attacks
        let mut buf = Vec::new();
        let bytes_read = reader
            .take((SNAPSHOT_DESERIALIZATION_BYTES_LIMIT + 1) as u64)
            .read_to_end(&mut buf)?;

        if bytes_read > SNAPSHOT_DESERIALIZATION_BYTES_LIMIT {
            return Err(SnapshotError::SizeLimitExceeded(
                SNAPSHOT_DESERIALIZATION_BYTES_LIMIT,
            ));
        }

        let buf = strip_layout_mark(&buf)?;

        // The last 8 bytes are the CRC, so we need to separate them
        if buf.len() < 8 {
            return Err(SnapshotError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File too short to contain CRC",
            )));
        }

        let (data_buf, _crc_buf) = buf.split_at(buf.len() - 8);
        let snapshot = Self::load_without_crc_check(data_buf)?;

        let computed_checksum = crc64(0, buf);
        // When we read the entire file, we also read the checksum into the buffer. The CRC has the
        // property that crc(0, buf.as_slice()) == 0 iff the last 8 bytes of buf are the checksum
        // of all the preceding bytes, and this is the property we are using here.
        if computed_checksum != 0 {
            return Err(SnapshotError::Crc64);
        }
        Ok(snapshot)
    }
}

impl<Data: Serialize> Snapshot<Data> {
    /// Saves `self` to the given [`Write`] instance, computing the CRC of the written data,
    /// and then writing the CRC into the `Write` instance, too, followed by the layout mark.
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), SnapshotError> {
        let mut crc_writer = CRC64Writer::new(writer);
        serialize(self, &mut crc_writer)?;
        // Write the CRC as raw bytes, not bitcode-serialized
        crc_writer
            .writer
            .write_all(&crc_writer.checksum().to_le_bytes())
            .map_err(SnapshotError::Io)?;
        crc_writer
            .writer
            .write_all(&layout_mark(FCVM_SNAPSHOT_LAYOUT))
            .map_err(SnapshotError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::MicrovmState;

    #[test]
    fn test_snapshot_restore() {
        let state = MicrovmState::default();
        let mut buf = Vec::new();

        Snapshot::new(state).save(&mut buf).unwrap();
        Snapshot::<MicrovmState>::load(&mut buf.as_slice()).unwrap();
    }

    #[test]
    fn test_parse_version_from_file() {
        use crate::persist::MicrovmState;
        let snapshot = Snapshot::new(MicrovmState::default());

        // Use a Vec<u8> that can grow as needed
        let mut snapshot_data = Vec::new();
        snapshot.save(&mut snapshot_data).unwrap();

        assert_eq!(
            get_format_version(&mut std::io::Cursor::new(&snapshot_data)).unwrap(),
            SNAPSHOT_VERSION
        );
    }

    #[test]
    fn test_bad_reader() {
        #[derive(Debug)]
        struct BadReader;

        impl Read for BadReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::InvalidInput.into())
            }
        }

        let mut reader = BadReader {};

        assert!(
            matches!(Snapshot::<()>::load(&mut reader), Err(SnapshotError::Io(inner)) if inner.kind() == std::io::ErrorKind::InvalidInput)
        );
    }

    #[test]
    fn test_bad_magic() {
        // Create a snapshot with corrupted magic and serialize it properly
        let mut bad_snapshot = Snapshot::new(());
        bad_snapshot.header.magic = 0xDEADBEEF;

        // Serialize the bad snapshot (without CRC for load_without_crc_check)
        let corrupted_data = bitcode::serialize(&bad_snapshot).unwrap();

        assert!(matches!(
            Snapshot::<()>::load_without_crc_check(&corrupted_data),
            Err(SnapshotError::InvalidMagic(_))
        ));
    }

    #[test]
    fn test_bad_crc() {
        let snapshot = Snapshot::new(());

        // Use a Vec<u8> that can grow as needed
        let mut valid_data = Vec::new();
        snapshot.save(&mut valid_data).unwrap();

        // Corrupt the CRC by changing the 8 bytes before the layout mark (where CRC is stored)
        let crc_end = valid_data.len() - LAYOUT_MARK_LEN;
        for byte in &mut valid_data[crc_end - 8..crc_end] {
            *byte ^= 0xFF; // Corrupt the CRC by flipping bits
        }

        assert!(matches!(
            Snapshot::<()>::load(&mut std::io::Cursor::new(&valid_data)),
            Err(SnapshotError::Crc64)
        ));
    }

    /// A file with no layout mark, which is every file an upstream build or an earlier build
    /// of the chain wrote, is refused by the mark before its bytes are decoded: the state here
    /// is one this build would load.
    #[test]
    fn test_a_snapshot_without_a_layout_mark_is_refused() {
        let mut data = Vec::new();
        Snapshot::new(MicrovmState::default())
            .save(&mut data)
            .unwrap();
        data.truncate(data.len() - LAYOUT_MARK_LEN);

        let err = Snapshot::<MicrovmState>::load(&mut data.as_slice()).unwrap_err();
        assert!(matches!(err, SnapshotError::NoLayoutMark), "{err}");
        assert!(err.to_string().starts_with("Invalid data version"), "{err}");
        assert!(matches!(
            get_format_version(&mut data.as_slice()),
            Err(SnapshotError::NoLayoutMark)
        ));
        // Shorter than a mark, and empty.
        for len in [LAYOUT_MARK_LEN - 1, 0] {
            assert!(matches!(
                Snapshot::<MicrovmState>::load(&mut &data[..len]),
                Err(SnapshotError::NoLayoutMark)
            ));
        }
    }

    /// A file of another layout is refused by its number, with state this build would load.
    #[test]
    fn test_a_snapshot_of_another_layout_is_refused() {
        let mut data = Vec::new();
        Snapshot::new(MicrovmState::default())
            .save(&mut data)
            .unwrap();
        let mark_at = data.len() - LAYOUT_MARK_LEN;
        let other = FCVM_SNAPSHOT_LAYOUT + 1;
        data[mark_at..].copy_from_slice(&layout_mark(other));

        let err = Snapshot::<MicrovmState>::load(&mut data.as_slice()).unwrap_err();
        assert!(
            matches!(err, SnapshotError::InvalidLayout(found, read)
                if found == other && read == FCVM_SNAPSHOT_LAYOUT),
            "{err}"
        );
        assert!(err.to_string().starts_with("Invalid data version"), "{err}");

        // A mark whose two copies of the number disagree is no mark.
        data[mark_at + 12] ^= 1;
        assert!(matches!(
            Snapshot::<MicrovmState>::load(&mut data.as_slice()),
            Err(SnapshotError::NoLayoutMark)
        ));
    }

    /// A build that does not know the mark (upstream, and every build of the chain before the
    /// mark) reads the last eight bytes of the file as the CRC of everything before them.
    /// Those are the mark's, so that check cannot pass: such a build refuses the file even if
    /// its decode of the bytes succeeds.
    #[test]
    fn test_a_reader_without_the_mark_fails_the_crc_of_a_marked_snapshot() {
        let mut data = Vec::new();
        Snapshot::new(MicrovmState::default())
            .save(&mut data)
            .unwrap();
        assert_ne!(crc64(0, &data), 0);
        // What such a build needs to see to accept the file.
        assert_eq!(crc64(0, &data[..data.len() - LAYOUT_MARK_LEN]), 0);
    }

    /// The layout number was chosen for this snapshot version. When upstream moves the
    /// version the saved state changed under the chain: give FCVM_SNAPSHOT_LAYOUT a number no
    /// earlier build used, and then update the version here.
    #[test]
    fn test_the_layout_was_chosen_for_this_snapshot_version() {
        assert_eq!(
            (SNAPSHOT_VERSION, FCVM_SNAPSHOT_LAYOUT),
            (Version::new(13, 0, 0), 1)
        );
    }

    #[test]
    fn test_bad_version() {
        // Different major version: shouldn't work
        let mut bad_snapshot = Snapshot::new(());
        bad_snapshot.header.version = Version::new(
            SNAPSHOT_VERSION.major() + 1,
            SNAPSHOT_VERSION.minor(),
            SNAPSHOT_VERSION.patch(),
        );
        let data = bitcode::serialize(&bad_snapshot).unwrap();

        assert!(matches!(
            Snapshot::<()>::load_without_crc_check(&data),
            Err(SnapshotError::InvalidFormatVersion(v)) if v.major() == SNAPSHOT_VERSION.major() + 1
        ));

        //  minor > SNAPSHOT_VERSION.minor: shouldn't work
        let mut bad_snapshot = Snapshot::new(());
        bad_snapshot.header.version = Version::new(
            SNAPSHOT_VERSION.major(),
            SNAPSHOT_VERSION.minor() + 1,
            SNAPSHOT_VERSION.patch(),
        );
        let data = bitcode::serialize(&bad_snapshot).unwrap();
        assert!(matches!(
            Snapshot::<()>::load_without_crc_check(&data),
            Err(SnapshotError::InvalidFormatVersion(v)) if v.minor() == SNAPSHOT_VERSION.minor() + 1
        ));

        // But we can support minor versions smaller or equal to ours. We also support
        // all patch versions within our supported major.minor version.
        let snapshot = Snapshot::new(());
        let data = bitcode::serialize(&snapshot).unwrap();
        Snapshot::<()>::load_without_crc_check(&data).unwrap();

        if SNAPSHOT_VERSION.minor() != 0 {
            let mut snapshot = Snapshot::new(());
            snapshot.header.version = Version::new(
                SNAPSHOT_VERSION.major(),
                SNAPSHOT_VERSION.minor() - 1,
                SNAPSHOT_VERSION.patch(),
            );

            let data = bitcode::serialize(&snapshot).unwrap();
            Snapshot::<()>::load_without_crc_check(&data).unwrap();
        }

        let mut snapshot = Snapshot::new(());
        snapshot.header.version =
            Version::new(SNAPSHOT_VERSION.major(), SNAPSHOT_VERSION.minor(), 0);
        let data = bitcode::serialize(&snapshot).unwrap();
        Snapshot::<()>::load_without_crc_check(&data).unwrap();

        let mut snapshot = Snapshot::new(());
        snapshot.header.version = Version::new(
            SNAPSHOT_VERSION.major(),
            SNAPSHOT_VERSION.minor(),
            SNAPSHOT_VERSION.patch() + 1,
        );
        let data = bitcode::serialize(&snapshot).unwrap();
        Snapshot::<()>::load_without_crc_check(&data).unwrap();

        let mut snapshot = Snapshot::new(());
        snapshot.header.version =
            Version::new(SNAPSHOT_VERSION.major(), SNAPSHOT_VERSION.minor(), 1024);
        let data = bitcode::serialize(&snapshot).unwrap();
        Snapshot::<()>::load_without_crc_check(&data).unwrap();
    }
}
