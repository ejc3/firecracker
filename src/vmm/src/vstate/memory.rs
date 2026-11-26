// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::ops::{Deref, Range};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use bitvec::vec::BitVec;
use kvm_bindings::{KVM_MEM_LOG_DIRTY_PAGES, kvm_userspace_memory_region};
use serde::{Deserialize, Serialize};
pub use vm_memory::bitmap::{AtomicBitmap, BS, Bitmap, BitmapSlice};
pub use vm_memory::mmap::MmapRegionBuilder;
use vm_memory::mmap::{MmapRegionError, NewBitmap};
pub use vm_memory::{
    Address, ByteValued, Bytes, FileOffset, GuestAddress, GuestMemoryRegion, GuestUsize,
    MemoryRegionAddress, MmapRegion, address,
};
use vm_memory::{
    GuestMemoryBackend, GuestMemoryError, GuestMemoryRegionBytes, VolatileSlice, WriteVolatile,
};

use crate::arch::host_page_size;
use crate::logger::{error, warn};
use crate::utils::{mib_to_bytes, u64_to_usize, usize_to_u64};
use crate::vmm_config::machine_config::HugePageConfig;
use crate::vstate::vm::{KvmVm, VmError};
use crate::{DirtyBitmap, align_down, align_up, warn_unrestricted};

/// Type of GuestMemoryMmap.
pub type GuestMemoryMmap = vm_memory::GuestRegionCollection<GuestRegionMmapExt>;

/// The alignment used to allocate guest memory.
/// Chosen to enable optimizations on host kernel, e.g. allow huge pages at the beginning of the memory space.
const GUEST_MEMORY_ALIGNMENT: usize = mib_to_bytes(2);
/// A mask to extract mmap's flags related to HUGETLB
const HUGETLB_FLAG_MASK: libc::c_int = libc::MAP_HUGETLB | (0x3F << libc::MAP_HUGE_SHIFT);

/// Size and alignment of the all-zero pages a Full dump into an empty file leaves as holes.
///
/// This is fcvm's `working_set::GRANULE` (src/uffd/working_set.rs in fcvm), the 4 KiB unit its
/// page server records and replays guest memory in, so every hole fcvm finds in a memory file
/// covers whole granules. It is not `host_page_size()`, which is larger on some arm64 hosts.
pub const HOLE_GRANULE: usize = 4096;

/// How much guest memory a Full dump into an empty file pages in and scans at a time. A run of
/// non-zero pages is written where the scan ends it or at the end of its chunk, so the chunk
/// bounds the size of each write and the time between populating a chunk and writing it.
const DUMP_CHUNK: usize = mib_to_bytes(2);

/// The most host pages in a [`DUMP_CHUNK`]: a host page is at least 4 KiB.
const DUMP_CHUNK_PAGES: usize = DUMP_CHUNK / 4096;

/// The bits of a /proc/self/pagemap entry that say a page has a page table entry: 63, the page is
/// in memory, and 62, it is swapped out. For small pages and transparent huge pages the kernel
/// sets 62 for every entry that is neither empty nor present: swap, migration, and markers. A
/// hugetlb entry that is not present (migration, hwpoison) gets neither bit, so this alone cannot
/// say that a page has no entry.
const PAGEMAP_PRESENT_OR_SWAPPED: u64 = 0b11 << 62;

/// What a Full dump does with an all-zero page of [`HOLE_GRANULE`] bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPages {
    /// Write it like any other page.
    Write,
    /// Seek over it and leave a hole. Correct only when the file holds no data under the page,
    /// because a page that is skipped keeps whatever the file held there before.
    Hole(AbsentPages),
}

/// What a Full dump that leaves holes does with a page of anonymous guest memory that has no
/// page table entry: one that was never written, or that a discard emptied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsentPages {
    /// Read it like any other page. Required when a userfaultfd handler supplies the missing
    /// pages of guest memory, so that such a page can hold something other than zeros.
    Read,
    /// Leave it as a hole without reading it. Reading it would map the shared zero page (or take
    /// a huge page from the pool on hugetlbfs), and the resumed guest's first write to the page
    /// would then be a copy-on-write. On arm64, each such copy drops every shadow stage-2
    /// mapping of a nested guest.
    Skip,
}

/// Whether every byte of `slice` is zero. Reads guest memory in place, eight bytes at a time
/// where aligned, and stops at the first non-zero byte.
fn is_zero<B: BitmapSlice>(slice: &VolatileSlice<'_, B>) -> bool {
    let guard = slice.ptr_guard();
    let ptr = guard.as_ptr();
    let len = slice.len();
    let head = ptr.align_offset(8).min(len);
    let words = (len - head) / 8;
    // SAFETY: every read is inside `slice`, which stays valid for reads while `guard` lives, and
    // the word reads start at an 8-byte aligned address. The reads are volatile, as for all guest
    // memory, because the guest and other processes can share it.
    unsafe {
        (0..head).all(|i| ptr.add(i).read_volatile() == 0)
            && (0..words).all(|i| ptr.add(head).cast::<u64>().add(i).read_volatile() == 0)
            && (head + words * 8..len).all(|i| ptr.add(i).read_volatile() == 0)
    }
}

/// Moves the writer `len` bytes forward without writing.
fn seek_forward<T: std::io::Seek>(writer: &mut T, len: usize) -> Result<(), MemoryError> {
    let offset = i64::try_from(len).map_err(|_| MemoryError::SlotSizeTooLarge)?;
    writer
        .seek(SeekFrom::Current(offset))
        .map_err(MemoryError::SeekError)?;
    Ok(())
}

/// Writes all of `slice` to the writer. A failure is [`MemoryError::WriteMemory`] in both
/// [`ZeroPages`] modes.
fn write_memory<T: WriteVolatile, B: BitmapSlice>(
    writer: &mut T,
    slice: &VolatileSlice<'_, B>,
) -> Result<(), MemoryError> {
    writer
        .write_all_volatile(slice)
        .map_err(|err| MemoryError::WriteMemory(err.into()))
}

/// Faults in every page of `slice` for reading with MADV_POPULATE_READ. A page the kernel cannot
/// provide at populate time (a hugetlb page the pool cannot supply, a file page past the end of a
/// truncated file or behind a read error) is then an error here instead of a SIGBUS when
/// [`is_zero`] reads it, which Firecracker's handler answers by exiting. The populate does not
/// hold the pages: if the file backing a mapping is truncated after this call and before the scan
/// of the chunk reads a page past the new end, that read still raises SIGBUS, where writing the
/// range with write(2), as [`ZeroPages::Write`] does, fails with EFAULT.
///
/// Before Linux 6.9 (commit 631426ba1d45, "make MADV_POPULATE_(READ|WRITE) handle
/// VM_FAULT_RETRY properly") the populate could retry forever on a page whose read fails, so on
/// an unpatched 5.14 to 6.8 kernel that case can hang here where write(2) fails with EFAULT.
///
/// Returns false on EINVAL, which means the advice is unknown (kernels before 5.14) or the
/// mapping does not allow it (a PROT_NONE range, for one).
fn populate_read<B: BitmapSlice>(slice: &VolatileSlice<'_, B>) -> Result<bool, MemoryError> {
    let guard = slice.ptr_guard();
    // SAFETY: the range lies inside `slice`, which stays mapped while `guard` lives.
    // MADV_POPULATE_READ only faults pages in for reading and changes no memory.
    let ret = unsafe {
        libc::madvise(
            guard.as_ptr().cast_mut().cast(),
            slice.len(),
            libc::MADV_POPULATE_READ,
        )
    };
    if ret == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EINVAL) {
        return Ok(false);
    }
    Err(MemoryError::WriteMemory(GuestMemoryError::IOError(err)))
}

/// Dumps `run`, at most [`DUMP_CHUNK`] bytes of a slot, onto the writer: pages it in with
/// [`populate_read`], writes each run of its pages of [`HOLE_GRANULE`] bytes that are not all
/// zero, and adds the zero pages to `skip_size`, the bytes to seek over before the next write.
/// Where MADV_POPULATE_READ returns EINVAL the run is written in full, as [`ZeroPages::Write`]
/// does, so that write(2) reports a page that cannot be read.
fn dump_nonzero_run<T: WriteVolatile + std::io::Seek, B: BitmapSlice>(
    writer: &mut T,
    run: &VolatileSlice<'_, B>,
    skip_size: &mut usize,
) -> Result<(), MemoryError> {
    let scan = populate_read(run)?;
    // The pending run of non-zero pages, as an offset and a length inside `run`.
    let mut write_start = 0;
    let mut write_size = 0;

    for offset in (0..run.len()).step_by(HOLE_GRANULE) {
        let page = run.subslice(offset, HOLE_GRANULE.min(run.len() - offset))?;
        if scan && is_zero(&page) {
            // We are at the end of a run of non-zero pages.
            if write_size > 0 {
                write_memory(writer, &run.subslice(write_start, write_size)?)?;
                write_size = 0;
            }
            *skip_size += page.len();
        } else {
            // We are at the end of a run of zero pages.
            if *skip_size > 0 {
                seek_forward(writer, *skip_size)?;
                *skip_size = 0;
            }
            if write_size == 0 {
                write_start = offset;
            }
            write_size += page.len();
        }
    }

    // Write the last run before the caller populates more memory, so no write covers more than
    // one chunk.
    if write_size > 0 {
        write_memory(writer, &run.subslice(write_start, write_size)?)?;
    }
    Ok(())
}

/// What a dump does with one page of guest memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageAction {
    /// Seek over it, so the file keeps what it holds there.
    Skip,
    /// Write zeros, without reading the page.
    Zeros,
    /// Read the page from guest memory and write it.
    Copy,
}

/// What a dump does with a page, given whether a write and a discard marked it since the marks
/// were last taken, and whether it has a page table entry ([`with_entries`]).
///
/// A discarded page is written as zeros, without reading it, when nothing wrote it since, or
/// when it has no page table entry: a page written before its discard has none, and a page
/// written after it has one. Every other written page is read and written. An unplugged slot
/// cannot be read, and the dump gives it no write marks, so its discarded pages are written as
/// zeros and the rest of it is seeked over.
fn page_action(written: bool, discarded: bool, has_entry: bool) -> PageAction {
    if discarded && !(written && has_entry) {
        PageAction::Zeros
    } else if written {
        PageAction::Copy
    } else {
        PageAction::Skip
    }
}

/// Which pages have a page table entry, as bit `i` for page `i`, given each page's
/// /proc/self/pagemap entry and its mincore(2) byte.
///
/// A page has no entry only when pagemap shows it neither present nor swapped and mincore reports
/// it not resident. Neither is enough alone. pagemap shows a small page that is swapped out or
/// migrating as swapped, and mincore reports it resident only while it is in the swap cache. A
/// hugetlb entry that is not present, such as a migration entry, gets neither pagemap bit, and
/// mincore reports it resident. Taking both needs no knowledge of the mapping, which matters
/// because a private-file discard leaves ordinary anonymous memory in a hugetlbfs region.
fn with_entries(pagemap: &[u64], resident: &[u8]) -> u64 {
    pagemap
        .iter()
        .zip(resident)
        .enumerate()
        .fold(0, |mapped, (page, (&entry, &resident))| {
            let has_entry = entry & PAGEMAP_PRESENT_OR_SWAPPED != 0 || resident & 1 != 0;
            mapped | (u64::from(has_entry) << page)
        })
}

/// State that one dump shares across the slots it writes.
#[derive(Debug, Default)]
struct DumpScratch {
    /// The zeros the dump writes from, allocated on first use.
    zeros: Vec<u8>,
    /// /proc/self/pagemap: `None` until first use, then the open file, or `None` when it cannot
    /// be read.
    pagemap: Option<Option<File>>,
}

impl DumpScratch {
    /// Writes `len` zeros.
    fn write_zeros<T: WriteVolatile>(
        &mut self,
        writer: &mut T,
        mut len: usize,
    ) -> Result<(), MemoryError> {
        if self.zeros.is_empty() {
            self.zeros = vec![0; DUMP_CHUNK];
        }
        while len > 0 {
            let chunk = len.min(self.zeros.len());
            writer.write_all_volatile(&VolatileSlice::from(&mut self.zeros[..chunk]))?;
            len -= chunk;
        }
        Ok(())
    }

    /// Which of the `count` host pages from host address `addr` have a page table entry, as bit
    /// `i % 64` of word `i / 64`, by [`with_entries`]. `count` is 1 to [`DUMP_CHUNK_PAGES`].
    ///
    /// /proc/self/pagemap is read with lseek(2) and read(2), which Firecracker's seccomp filters
    /// allow on the thread that dumps, as they do mincore(2). When pagemap or mincore cannot be
    /// read every bit is set, so the caller reads those pages.
    fn mapped(
        &mut self,
        addr: usize,
        count: usize,
        page_size: usize,
    ) -> [u64; DUMP_CHUNK_PAGES / 64] {
        let all = [u64::MAX; DUMP_CHUNK_PAGES / 64];
        let mut entries = [0u8; DUMP_CHUNK_PAGES * 8];
        let entries = &mut entries[..count * 8];
        let pagemap = self.pagemap.get_or_insert_with(|| {
            File::open("/proc/self/pagemap")
                .inspect_err(|err| {
                    warn!(
                        "Cannot open /proc/self/pagemap ({err}), so a snapshot reads the guest \
                         pages it would otherwise leave unread"
                    )
                })
                .ok()
        });
        let Some(file) = pagemap else {
            return all;
        };
        let offset = usize_to_u64(addr / page_size * 8);
        if let Err(err) = file
            .seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(entries))
        {
            warn!(
                "Cannot read /proc/self/pagemap ({err}), so a snapshot reads the guest pages it \
                 would otherwise leave unread"
            );
            self.pagemap = Some(None);
            return all;
        }
        let mut resident = [0u8; DUMP_CHUNK_PAGES];
        // SAFETY: `addr` is page aligned, and the `count` pages from it are guest memory, which
        // stays mapped while the dump runs. `resident` has a byte for each of them.
        let ret = unsafe {
            libc::mincore(
                addr as *mut c_void,
                count * page_size,
                resident.as_mut_ptr(),
            )
        };
        if ret != 0 {
            warn!(
                "mincore failed ({}), so a snapshot reads these guest pages",
                io::Error::last_os_error()
            );
            return all;
        }
        let mut mapped = [0u64; DUMP_CHUNK_PAGES / 64];
        for ((word, entries), resident) in mapped
            .iter_mut()
            .zip(entries.chunks(64 * 8))
            .zip(resident[..count].chunks(64))
        {
            let mut pagemap = [0u64; 64];
            for (entry, bytes) in pagemap.iter_mut().zip(entries.chunks_exact(8)) {
                *entry = u64::from_ne_bytes(bytes.try_into().unwrap());
            }
            *word = with_entries(&pagemap[..resident.len()], resident);
        }
        mapped
    }
}

/// Writes one slot of a dump as runs of adjacent pages with the same [`PageAction`]: one seek,
/// one write of zeros or one write from guest memory per run.
struct SlotWriter<'s, 'a, T> {
    slot: &'s GuestMemorySlot<'a>,
    writer: &'s mut T,
    scratch: &'s mut DumpScratch,
    /// The pending run: its action, and its offset and length in the slot.
    action: PageAction,
    start: usize,
    len: usize,
}

impl<T: WriteVolatile + std::io::Seek> SlotWriter<'_, '_, T> {
    /// Adds `len` bytes at `offset` in the slot to the pending run, writing the run first when
    /// the action changes.
    fn push(&mut self, offset: usize, len: usize, action: PageAction) -> Result<(), MemoryError> {
        if self.len > 0 && self.action != action {
            self.flush()?;
        }
        if self.len == 0 {
            self.action = action;
            self.start = offset;
        }
        self.len += len;
        Ok(())
    }

    /// Writes the pending run.
    fn flush(&mut self) -> Result<(), MemoryError> {
        match self.action {
            PageAction::Skip => seek_forward(self.writer, self.len)?,
            PageAction::Zeros => self.scratch.write_zeros(self.writer, self.len)?,
            PageAction::Copy => self
                .writer
                .write_all_volatile(&self.slot.slice.subslice(self.start, self.len)?)?,
        }
        self.len = 0;
        Ok(())
    }
}

/// Errors associated with dumping guest memory to file.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum MemoryError {
    /// Cannot dump memory: {0}
    WriteMemory(GuestMemoryError),
    /// Cannot create mmap region: {0}
    MmapRegionError(MmapRegionError),
    /// Cannot create guest memory
    VmMemoryError,
    /// Cannot create memfd: {0}
    Memfd(memfd::Error),
    /// Cannot resize memfd file: {0}
    MemfdSetLen(std::io::Error),
    /// Total sum of memory regions exceeds largest possible file offset
    OffsetTooLarge,
    /// Memory region has zero size
    ZeroSize,
    /// Guest memory size of {0} bytes exceeds the architectural maximum of {1} bytes
    GuestMemoryTooLarge(usize, usize),
    /// Memory region has zero slots
    ZeroSlots,
    /// Memory region of {region_size} bytes is not evenly divisible into {slot_count} slots
    Unaligned {
        /// Region size in bytes.
        region_size: u64,
        /// Number of slots declared in the snapshot.
        slot_count: usize,
    },
    /// Error protecting memory slot: {0}
    Mprotect(std::io::Error),
    /// Error calling madvise: {0}
    Madvise(std::io::Error),
    /// Size too large for i64 conversion
    SlotSizeTooLarge,
    /// Seek error: {0}
    SeekError(std::io::Error),
    /// A discard of unplugged guest memory failed, so a diff snapshot cannot say what that memory holds. Take a full snapshot first.
    FailedDiscardOfUnpluggedMemory,
    /// Volatile memory error: {0}
    VolatileMemoryError(vm_memory::VolatileMemoryError),
}

impl From<vm_memory::VolatileMemoryError> for MemoryError {
    fn from(e: vm_memory::VolatileMemoryError) -> Self {
        MemoryError::VolatileMemoryError(e)
    }
}

/// Type of the guest region
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum GuestRegionType {
    /// Guest DRAM
    Dram,
    /// Hotpluggable memory
    Hotpluggable,
}

/// A region of guest memory. Unmapped on drop.
/// This struct doesn't provide any facility to access the memory or read its properties, see [GuestRegionMmap].
#[derive(Debug)]
struct RawGuestRegionMmap {
    mmap_address: *mut u8,
    mmap_size: usize,
}

impl RawGuestRegionMmap {
    /// Allocates a `PROT_NONE` region of memory, aligned to [`GUEST_MEMORY_ALIGNMENT`] (2 MiB).
    ///
    /// To actually use the memory, callers must either re-map it (e.g. with `MAP_FIXED`) or use
    /// [`Self::allocate`] which does this automatically.
    ///
    /// # Size rounding
    ///
    /// The returned `mmap_size` may be **larger** than `requested_size`: it is rounded up to the
    /// nearest multiple of the page size used for the mapping. For regular mappings this is the
    /// host page size (typically 4 KiB); for hugetlb mappings it is the huge page size (2 MiB).
    /// This is because `mmap`/`munmap` operate at page granularity — the kernel cannot map or
    /// protect a partial page. For example, requesting 5000 bytes on a 4 KiB-page host yields
    /// an `mmap_size` of 8192 bytes (2 pages).
    ///
    /// The invariant is: `mmap_size == align_up!(requested_size, page_size)`.
    fn allocate_protected(
        requested_size: usize,
        hugetlb_flags: libc::c_int,
    ) -> Result<Self, MemoryError> {
        // Strategy: over-allocate by up to `GUEST_MEMORY_ALIGNMENT - page_size`, then trim
        // the unaligned head and tail via munmap, leaving a 2 MiB-aligned region of
        // `size_page_multiple` bytes.
        //
        //         ptr                          ptr + alloc_size
        //          |                                  |
        //          v                                  v
        //          +------+------------------+--------+
        //          | head |   guest memory   |  tail  |
        //          +------+------------------+--------+
        //                 ^                  ^
        //                 |                  |
        //           aligned_ptr       aligned_ptr + size_page_multiple
        //          (2 MiB-aligned)
        //
        //  1. Round requested_size up to page size -> `size_page_multiple`.
        //  2. mmap(`size_page_multiple + GUEST_MEMORY_ALIGNMENT - page_size`) -> `ptr`.
        //     `ptr` is page-aligned, so the worst-case distance to the next 2 MiB boundary
        //     is `GUEST_MEMORY_ALIGNMENT - page_size` bytes. This guarantees the allocation
        //     contains at least one 2 MiB-aligned region of `size_page_multiple` bytes.
        //  3. Compute `aligned_ptr` = first 2 MiB boundary ≥ `ptr`.
        //  4. munmap the head (`ptr..aligned_ptr`) and tail (`aligned_ptr + size_page_multiple..end`).
        //  5. Return the 2 MiB-aligned region of `size_page_multiple` bytes.
        //
        // Note: when `hugetlb_flags` is set, `page_size` equals `GUEST_MEMORY_ALIGNMENT` (2 MiB),
        // so `alloc_size == size_page_multiple` (no over-allocation). This is correct because the
        // kernel guarantees hugetlb mappings are already aligned to the huge page size.
        if requested_size == 0 {
            return Err(MemoryError::ZeroSize);
        }
        let page_size = Self::page_size(hugetlb_flags);
        let size_page_multiple = align_up!(requested_size, page_size);

        // Over-allocate to guarantee we can find a 2 MiB-aligned sub-region of the desired size.
        // The over-allocation is `GUEST_MEMORY_ALIGNMENT - page_size` because the mmap return
        // address is page-aligned, so we need at most that many extra bytes to reach the next
        // 2 MiB boundary. For hugetlb (page_size == 2 MiB), this is zero — no over-allocation
        // needed since the kernel already returns 2 MiB-aligned addresses.
        let alloc_size = size_page_multiple + GUEST_MEMORY_ALIGNMENT - page_size;

        // SAFETY: anonymous private mapping with no fd; does not alias existing memory.
        // The returned region is PROT_NONE (inaccessible) until the caller re-maps it.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc_size,
                libc::PROT_NONE,
                hugetlb_flags | libc::MAP_PRIVATE | libc::MAP_NORESERVE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(MemoryError::MmapRegionError(MmapRegionError::Mmap(
                std::io::Error::last_os_error(),
            )));
        }

        // Find the first 2 MiB-aligned address within the allocation.
        let aligned_ptr = align_up!(ptr as usize, GUEST_MEMORY_ALIGNMENT);

        // Compute head/tail sizes to trim. Both are guaranteed to be page-aligned because
        // `ptr` is page-aligned (from mmap), `aligned_ptr` is 2 MiB-aligned (a multiple of
        // page size), and `size_page_multiple` is a page multiple by construction.
        let head_size = aligned_ptr - ptr as usize;
        let tail_size = alloc_size - head_size - size_page_multiple;

        // Sanity checks: the three parts must exactly partition the allocation.
        assert_eq!(
            head_size + size_page_multiple + tail_size,
            alloc_size,
            "Allocated memory is not fully partitioned"
        );
        assert_eq!(
            ptr as usize + head_size,
            aligned_ptr,
            "Head and aligned portion aren't adjacent"
        );
        assert_eq!(
            aligned_ptr + size_page_multiple,
            ptr as usize + alloc_size - tail_size,
            "Tail and aligned portion aren't adjacent"
        );
        assert_eq!(
            head_size % page_size,
            0,
            "head size is not a multiple of page-size"
        );
        assert_eq!(
            tail_size % page_size,
            0,
            "tail size is not a multiple of page-size"
        );

        // Unmap the head and tail to release them back to the kernel.
        // Failures are non-critical (at worst we leak a `PROT_NONE` VMA slot), so we only print a warning.
        if head_size > 0 {
            // SAFETY: `ptr` is page-aligned and `head_size` is a page multiple within our mapping.
            let ret = unsafe { libc::munmap(ptr, head_size) };
            if ret != 0 {
                warn_unrestricted!("munmap failed: {}", std::io::Error::last_os_error(),);
            }
        }
        if tail_size > 0 {
            // SAFETY: address is page-aligned and `tail_size` is a page multiple within our mapping.
            let ret = unsafe {
                libc::munmap((aligned_ptr + size_page_multiple) as *mut c_void, tail_size)
            };
            if ret != 0 {
                warn_unrestricted!("munmap failed: {}", std::io::Error::last_os_error(),);
            }
        }

        Ok(Self {
            mmap_address: aligned_ptr as *mut u8,
            mmap_size: size_page_multiple,
        })
    }

    /// Allocates a region of memory of the given size, aligned with [GUEST_MEMORY_ALIGNMENT].
    ///
    /// Note: the returned region of memory might be bigger than the requested size, as it is aligned to page-size.
    fn allocate(
        size: usize,
        prot: libc::c_int,
        flags: libc::c_int,
        fd: libc::c_int,
        offset: libc::off_t,
    ) -> Result<Self, MemoryError> {
        let hugetlb_flags = Self::extract_huge_tlb_mmap_flags(flags);

        // This function works by doing a mmap fixed over a GuestMmapRegion allocated with PROT_NONE.
        let ret = Self::allocate_protected(size, hugetlb_flags)?;

        // SAFETY: this mmap is performed over the same memory region we just allocated, so it's
        // guarantee it's not in use elsewhere.
        let addr = unsafe {
            libc::mmap(
                ret.mmap_address.cast::<c_void>(),
                ret.mmap_size,
                prot,
                flags | libc::MAP_FIXED,
                fd,
                offset,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(MemoryError::MmapRegionError(MmapRegionError::Mmap(
                std::io::Error::last_os_error(),
            )));
        }
        assert_eq!(
            addr.cast::<u8>(),
            ret.mmap_address,
            "mmap fixed returned a different address"
        );

        Ok(ret)
    }

    /// Given some mmap flags, extracts the ones related to hugetlb
    fn extract_huge_tlb_mmap_flags(flags: libc::c_int) -> libc::c_int {
        flags & HUGETLB_FLAG_MASK
    }

    fn page_size(flags: libc::c_int) -> usize {
        match Self::extract_huge_tlb_mmap_flags(flags) {
            0 => host_page_size(),
            flags if flags == libc::MAP_HUGETLB | libc::MAP_HUGE_2MB => mib_to_bytes(2),
            _ => unreachable!("Only 2MiB hugetlb pages are supported"),
        }
    }
}

impl Drop for RawGuestRegionMmap {
    fn drop(&mut self) {
        // SAFETY: All memory access is done via GuestRegionMmap, and at this point there are no references to that.
        let ret = unsafe { libc::munmap(self.mmap_address.cast::<c_void>(), self.mmap_size) };
        // Panicking if unmapping failed: otherwise the guest memory might still be accessible after drop!
        assert_eq!(
            ret,
            0,
            "munmap of guest region failed: {}",
            std::io::Error::last_os_error()
        );
    }
}

/// SAFETY: Send and Sync aren't automatically inherited for the raw address pointer.
/// RawGuestRegionMmap exclusively owns the memory region and doesn't expose the raw pointer for concurrent mutation.
unsafe impl Send for RawGuestRegionMmap {}
/// SAFETY: See comment above.
unsafe impl Sync for RawGuestRegionMmap {}

/// A region of guest memory. Unmapped on drop.
/// Note: this implements automatic defer for vm_memory::GuestRegionMmap, so it can be used in its place.
#[derive(Debug)]
pub struct GuestRegionMmap {
    /// Held for its `Drop` impl which unmaps the guest memory region.
    _region: RawGuestRegionMmap,
    proxy: vm_memory::GuestRegionMmap<Option<AtomicBitmap>>,
}

impl Deref for GuestRegionMmap {
    type Target = vm_memory::GuestRegionMmap<Option<AtomicBitmap>>;

    fn deref(&self) -> &Self::Target {
        &self.proxy
    }
}

impl GuestRegionMmap {
    /// Creates a new [GuestRegionMmap], by actually allocating the guest memory with the given properties.
    pub(crate) fn allocate(
        guest_base: GuestAddress,
        size: usize,
        prot: libc::c_int,
        flags: libc::c_int,
        file_offset: Option<FileOffset>,
        track_dirty_pages: bool,
    ) -> Result<Self, MemoryError> {
        let (fd, offset) = if let Some(ref f_off) = file_offset {
            let fd = f_off.file().as_raw_fd();
            let offset = f_off
                .start()
                .try_into()
                .map_err(|_| MemoryError::OffsetTooLarge)?;
            (fd, offset)
        } else {
            (-1, 0)
        };
        let region = RawGuestRegionMmap::allocate(size, prot, flags, fd, offset)?;

        let mut builder = MmapRegionBuilder::new_with_bitmap(
            size,
            track_dirty_pages.then(|| AtomicBitmap::with_len(size)),
        )
        .with_mmap_prot(prot)
        .with_mmap_flags(flags);

        if let Some(file_offset) = file_offset {
            builder = builder.with_file_offset(file_offset);
        }

        // SAFETY: the memory mapping is valid, and has the same flags passed to builder.
        unsafe {
            builder = builder.with_raw_mmap_pointer(region.mmap_address);
        }

        let mmap_region = builder.build().map_err(MemoryError::MmapRegionError)?;
        // The memory cannot be double-owned: OwnedRegionMmap should be responsible for deallocation
        assert!(!mmap_region.owned());
        // `region` and `mapping` need to point to the same address.
        assert_eq!(region.mmap_address, mmap_region.as_ptr());
        // Size might differ, as region.mmap_size is page-aligned. But it for sure cannot be smaller.
        assert!(region.mmap_size >= mmap_region.size());

        Ok(GuestRegionMmap {
            _region: region,
            proxy: vm_memory::GuestRegionMmap::new(mmap_region, guest_base)
                .ok_or(MemoryError::VmMemoryError)?,
        })
    }
}

/// An extension to GuestMemoryRegion that can be split into multiple KVM slots of
/// the same slot_size, and stores the type of region, and the starting KVM slot number.
#[derive(Debug)]
pub struct GuestRegionMmapExt {
    /// the wrapped GuestRegionMmap
    pub inner: GuestRegionMmap,
    /// the type of region
    pub region_type: GuestRegionType,
    /// the starting KVM slot number assigned to this region
    pub slot_from: u32,
    /// the size of the slots of this region. Hotpluggable slots are a multiple of 64 host pages
    /// (256 KiB with 4 KiB pages), so each starts on a word of the region's dirty bitmap.
    slot_size: usize,
    /// a bitvec indicating whether slot `i` is plugged into KVM (1) or not (0)
    pub plugged: Mutex<BitVec>,
    /// With dirty tracking on, the pages that a discard emptied since a snapshot last took
    /// them. A discard changes guest memory without a guest write, so KVM's dirty log never has
    /// it.
    discarded: Option<AtomicBitmap>,
    /// Set when a discard failed in a range that holds an unplugged slot. Such a discard can
    /// have freed some pages of the range and not others, nothing recorded which, and a snapshot
    /// cannot read an unplugged slot to find out. A Diff snapshot is refused while it is set
    /// ([`Self::failed_discard_of_unplugged_memory`]).
    failed_unplugged_discard: AtomicBool,
    /// Whether the region maps a file privately on a filesystem other than hugetlbfs
    /// ([`Self::maps_file_privately`]), read once when the region is made.
    private_file: bool,
}

/// What a discard does with a backing page its range covers only part of. Only memory on huge
/// pages has such pages: a discard is made of host pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialPages {
    /// Writes zeros over the part in the range, whatever that costs: writing to a huge page
    /// that is not backed yet allocates it, and writing to one that a private mapping shares
    /// with its file copies it.
    Zero,
    /// Writes zeros over the part in the range only where the write allocates nothing: the
    /// huge page is in memory and is this mapping's own. A huge page that is absent, or that a
    /// private mapping still shares with its file, is left as it is, and the part in the
    /// range keeps the bytes it had.
    ZeroWithoutAllocating,
}

/// Whether a write to a page with this /proc/self/pagemap entry allocates a backing page: the
/// page is not in memory (bit 63 clear), or it is a page of the file (bit 61) under a private
/// mapping, which a write copies.
fn pagemap_entry_write_allocates(entry: u64, shared: bool) -> bool {
    const PRESENT: u64 = 1 << 63;
    const FILE_PAGE: u64 = 1 << 61;
    entry & PRESENT == 0 || (!shared && entry & FILE_PAGE != 0)
}

/// Whether a write to the page at `addr` of a mapping would allocate a backing page
/// ([`pagemap_entry_write_allocates`]). `None` when /proc/self/pagemap cannot be read.
fn write_allocates(addr: usize, shared: bool) -> Option<bool> {
    use std::os::unix::fs::FileExt;
    static PAGEMAP: OnceLock<Option<File>> = OnceLock::new();
    let pagemap = PAGEMAP
        .get_or_init(|| {
            File::open("/proc/self/pagemap")
                .inspect_err(|err| {
                    warn!(
                        "Cannot open /proc/self/pagemap ({err}), so a balloon discard zeroes \
                         every partial huge page"
                    )
                })
                .ok()
        })
        .as_ref()?;
    let mut entry = [0u8; 8];
    pagemap
        .read_exact_at(&mut entry, (addr / host_page_size() * 8) as u64)
        .ok()?;
    Some(pagemap_entry_write_allocates(
        u64::from_ne_bytes(entry),
        shared,
    ))
}

/// Whether `region` maps a file privately on a filesystem other than hugetlbfs. It takes an
/// fstatfs(2), which the seccomp filters of a running VMM do not list, so a region asks when
/// it is made: a region with a file is made before the filters are installed.
fn maps_file_privately(region: &GuestRegionMmap) -> bool {
    match region.file_offset() {
        Some(file_offset) if region.flags() & libc::MAP_PRIVATE != 0 => {
            BackingFileSystem::of(file_offset.file())
                .is_ok_and(|fs| fs.hugetlbfs_page_size().is_none())
        }
        _ => false,
    }
}

/// The bitmap of a region's discarded pages, present when the region tracks dirty pages or maps
/// a file privately. Without dirty tracking a private file mapping has its written pages read
/// from /proc/self/pagemap, which cannot see a discard either.
fn discard_bitmap(region: &GuestRegionMmap) -> Option<AtomicBitmap> {
    let private_file = region.file_offset().is_some() && region.flags() & libc::MAP_PRIVATE != 0;
    (region.bitmap().is_some() || private_file)
        .then(|| AtomicBitmap::with_len(u64_to_usize(region.len())))
}

/// A guest memory slot, which is a slice of a guest memory region
#[derive(Debug)]
pub struct GuestMemorySlot<'a> {
    /// KVM memory slot number
    pub(crate) slot: u32,
    /// Start guest address of the slot
    pub(crate) guest_addr: GuestAddress,
    /// Corresponding slice in host memory
    pub(crate) slice: VolatileSlice<'a, BS<'a, Option<AtomicBitmap>>>,
}

impl From<&GuestMemorySlot<'_>> for kvm_userspace_memory_region {
    fn from(mem_slot: &GuestMemorySlot) -> Self {
        let flags = if mem_slot.slice.bitmap().is_some() {
            KVM_MEM_LOG_DIRTY_PAGES
        } else {
            0
        };
        kvm_userspace_memory_region {
            flags,
            slot: mem_slot.slot,
            guest_phys_addr: mem_slot.guest_addr.raw_value(),
            memory_size: mem_slot.slice.len() as u64,
            userspace_addr: mem_slot.slice.ptr_guard().as_ptr() as u64,
        }
    }
}

impl<'a> GuestMemorySlot<'a> {
    /// Dumps the marked pages of this slot onto the writer and seeks over the rest.
    ///
    /// `dirty_bitmap` has the pages written since the marks were last taken, and is `None` for
    /// an unplugged slot, which cannot be read. `discarded` has the pages a discard emptied in
    /// that time. [`page_action`] decides what each page gets.
    // Inlined into a large caller, this per-page loop runs out of registers and slows down.
    #[inline(never)]
    fn dump_dirty<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        dirty_bitmap: Option<&[u64]>,
        discarded: &[u64],
        page_size: usize,
        scratch: &mut DumpScratch,
    ) -> Result<(), MemoryError> {
        let num_pages = self.slice.len() / page_size;
        assert_eq!(discarded.len(), num_pages.div_ceil(64));
        if let Some(dirty_bitmap) = dirty_bitmap {
            assert_eq!(dirty_bitmap.len(), discarded.len());
        }
        let host = self.slice.ptr_guard().as_ptr() as usize;
        let mut out = SlotWriter {
            slot: self,
            writer,
            scratch,
            action: PageAction::Skip,
            start: 0,
            len: 0,
        };

        for (word, &discarded) in discarded.iter().enumerate() {
            let first = word * 64;
            // We process 64 pages at a time, however the number of pages in the slot might not
            // be a multiple of 64, and the last word can have bits past the last page.
            let count = (num_pages - first).min(64);
            let in_slot = u64::MAX >> (64 - count);
            let written = dirty_bitmap.map_or(0, |bitmap| bitmap[word]) & in_slot;
            let discarded = discarded & in_slot;
            if written | discarded == 0 {
                out.push(first * page_size, count * page_size, PageAction::Skip)?;
                continue;
            }
            // A discarded page with a write mark was written before its discard or after it.
            let mapped = if discarded & written == 0 {
                0
            } else {
                out.scratch
                    .mapped(host + first * page_size, count, page_size)[0]
            };
            for page in 0..count {
                let bit = |bits: u64| bits & (1 << page) != 0;
                let action = page_action(bit(written), bit(discarded), bit(mapped));
                out.push((first + page) * page_size, page_size, action)?;
            }
        }

        // Write the last run, a seek included, so that the next slot starts writing at the
        // correct offset.
        out.flush()
    }

    /// Dumps this slot onto the writer, seeking over each all-zero page of [`HOLE_GRANULE`]
    /// bytes instead of writing it. Non-zero runs go from guest memory straight to the writer.
    ///
    /// The slot goes [`DUMP_CHUNK`] bytes at a time: each chunk is paged in with
    /// MADV_POPULATE_READ, scanned, and has its non-zero runs written before the next chunk is
    /// read. Given `absent`, the host pages of a chunk that have no page table entry
    /// ([`DumpScratch::mapped`]) are seeked over without being paged in or read, and each run of
    /// pages between them is paged in and scanned on its own. That is correct only for anonymous
    /// memory whose missing pages no userfaultfd handler supplies, where such a page reads as
    /// zeros.
    fn dump_nonzero<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        mut absent: Option<&mut DumpScratch>,
    ) -> Result<(), MemoryError> {
        let len = self.slice.len();
        let page_size = host_page_size();
        let host = self.slice.ptr_guard().as_ptr() as usize;
        // Bytes of zero pages to seek over before the next write. A zero run can span chunks.
        let mut skip_size = 0;

        for chunk_start in (0..len).step_by(DUMP_CHUNK) {
            let chunk_len = DUMP_CHUNK.min(len - chunk_start);
            let Some(scratch) = absent.as_deref_mut() else {
                let chunk = self.slice.subslice(chunk_start, chunk_len)?;
                dump_nonzero_run(writer, &chunk, &mut skip_size)?;
                continue;
            };
            let pages = chunk_len.div_ceil(page_size);
            let mapped = scratch.mapped(host + chunk_start, pages, page_size);
            let has_entry = |page: usize| mapped[page / 64] & (1 << (page % 64)) != 0;
            let mut page = 0;
            while page < pages {
                let run_pages = (page..pages)
                    .take_while(|&next| has_entry(next) == has_entry(page))
                    .count();
                let run_start = page * page_size;
                let run_len = (run_pages * page_size).min(chunk_len - run_start);
                if has_entry(page) {
                    let run = self.slice.subslice(chunk_start + run_start, run_len)?;
                    dump_nonzero_run(writer, &run, &mut skip_size)?;
                } else {
                    skip_size += run_len;
                }
                page += run_pages;
            }
        }

        // Seek over a trailing zero run too, so that the next slot starts writing at the correct
        // offset.
        if skip_size > 0 {
            seek_forward(writer, skip_size)?;
        }

        Ok(())
    }

    /// Makes the slot host memory PROT_NONE (true) or PROT_READ|PROT_WRITE (false)
    pub(crate) fn protect(&self, protected: bool) -> Result<(), MemoryError> {
        let prot = if protected {
            libc::PROT_NONE
        } else {
            libc::PROT_READ | libc::PROT_WRITE
        };
        // SAFETY: Parameters refer to an existing host memory region
        let ret = unsafe {
            libc::mprotect(
                self.slice.ptr_guard_mut().as_ptr().cast(),
                self.slice.len(),
                prot,
            )
        };
        if ret != 0 {
            Err(MemoryError::Mprotect(std::io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }
}

impl GuestRegionMmapExt {
    /// Whether the region maps a file privately on a filesystem other than hugetlbfs, as a
    /// restore from a snapshot file does. A page the guest writes in it becomes a private
    /// anonymous copy, which /proc/self/pagemap tells from one it only read.
    pub(crate) fn maps_file_privately(&self) -> bool {
        self.private_file
    }

    /// Whether the mapping is `MAP_SHARED`, so another mapping of the same
    /// file sees the same pages.
    pub fn is_shared(&self) -> bool {
        self.inner.flags() & libc::MAP_SHARED != 0
    }

    /// Adds a DRAM region which only contains a single plugged slot
    pub(crate) fn dram_from_mmap_region(region: GuestRegionMmap, slot: u32) -> Self {
        let slot_size = u64_to_usize(region.len());
        GuestRegionMmapExt {
            discarded: discard_bitmap(&region),
            private_file: maps_file_privately(&region),
            failed_unplugged_discard: AtomicBool::new(false),
            inner: region,
            region_type: GuestRegionType::Dram,
            slot_from: slot,
            slot_size,
            plugged: Mutex::new(BitVec::repeat(true, 1)),
        }
    }

    /// Adds an hotpluggable region which can contain multiple slots and is initially unplugged
    pub(crate) fn hotpluggable_from_mmap_region(
        region: GuestRegionMmap,
        slot_from: u32,
        slot_size: usize,
    ) -> Self {
        let slot_cnt = (u64_to_usize(region.len())) / slot_size;
        assert!(
            slot_size.is_multiple_of(64 * host_page_size()),
            "hotpluggable slot size {slot_size} is not a multiple of 64 host pages"
        );

        GuestRegionMmapExt {
            discarded: discard_bitmap(&region),
            private_file: maps_file_privately(&region),
            failed_unplugged_discard: AtomicBool::new(false),
            inner: region,
            region_type: GuestRegionType::Hotpluggable,
            slot_from,
            slot_size,
            plugged: Mutex::new(BitVec::repeat(false, slot_cnt)),
        }
    }

    pub(crate) fn from_state(
        region: GuestRegionMmap,
        state: &GuestMemoryRegionState,
        slot_from: u32,
    ) -> Result<Self, MemoryError> {
        let slot_cnt = state.plugged.len();
        let region_len = u64_to_usize(region.len());
        let slot_size = region_len
            .checked_div(slot_cnt)
            .ok_or(MemoryError::ZeroSlots)?;
        if slot_size * slot_cnt != region_len {
            return Err(MemoryError::Unaligned {
                region_size: region.len(),
                slot_count: slot_cnt,
            });
        }

        Ok(GuestRegionMmapExt {
            discarded: discard_bitmap(&region),
            private_file: maps_file_privately(&region),
            failed_unplugged_discard: AtomicBool::new(false),
            inner: region,
            slot_size,
            region_type: state.region_type,
            slot_from,
            plugged: Mutex::new(BitVec::from_iter(state.plugged.iter())),
        })
    }

    /// Check whether the given guest address range falls within plugged slots.
    pub(crate) fn check_range_plugged(
        &self,
        caddr: MemoryRegionAddress,
        len: usize,
    ) -> Result<(), GuestMemoryError> {
        // caddr is guaranteed to be within the region by the caller
        // (try_for_each_region_in_range and discard_range validate this).
        let from = self
            .start_addr()
            .checked_add(caddr.raw_value())
            .expect("caddr should be within the region");
        if self
            .slots_intersecting_range(from, len)
            .any(|(_, plugged)| !plugged)
        {
            return Err(GuestMemoryError::HostAddressNotAvailable);
        }
        Ok(())
    }

    pub(crate) fn slot_cnt(&self) -> u32 {
        u32::try_from(u64_to_usize(self.len()) / self.slot_size).unwrap()
    }

    /// The words of slot `index` in a bitmap of the whole region, one bit per page.
    pub(crate) fn slot_words(&self, index: usize) -> Range<usize> {
        let pages = self.slot_size / host_page_size();
        assert_eq!(index * pages % 64, 0);
        let first = index * pages / 64;
        first..first + pages.div_ceil(64)
    }

    pub(crate) fn mem_slot(&self, slot: u32) -> GuestMemorySlot<'_> {
        assert!(slot >= self.slot_from && slot < self.slot_from + self.slot_cnt());

        let offset = ((slot - self.slot_from) as u64) * (self.slot_size as u64);

        GuestMemorySlot {
            slot,
            guest_addr: self.start_addr().unchecked_add(offset),
            slice: self
                .inner
                .get_slice(MemoryRegionAddress(offset), self.slot_size)
                .expect("slot range should be valid"),
        }
    }

    /// Returns a snapshot of the slots and their state at the time of calling
    ///
    /// Note: to avoid TOCTOU races use only within VMM thread.
    pub(crate) fn slots(&self) -> impl Iterator<Item = (GuestMemorySlot<'_>, bool)> {
        self.plugged
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, b)| {
                (
                    self.mem_slot(self.slot_from + u32::try_from(i).unwrap()),
                    *b,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Returns a snapshot of the plugged slots at the time of calling
    ///
    /// Note: to avoid TOCTOU races use only within VMM thread.
    pub(crate) fn plugged_slots(&self) -> impl Iterator<Item = GuestMemorySlot<'_>> {
        self.slots()
            .filter(|(_, plugged)| *plugged)
            .map(|(slot, _)| slot)
    }

    pub(crate) fn slots_intersecting_range(
        &self,
        from: GuestAddress,
        len: usize,
    ) -> impl Iterator<Item = (GuestMemorySlot<'_>, bool)> {
        self.slots().filter(move |(slot, _)| {
            // Two intervals [a, b) and [c, d) intersect iff a < d && c < b.
            // This correctly handles the containment case where the slot fully
            // contains the range (or vice versa).
            let slot_start = slot.guest_addr;
            let Some(slot_end) = slot_start.checked_add(slot.slice.len() as u64) else {
                return false;
            };
            let Some(range_end) = from.checked_add(len as u64) else {
                return false;
            };
            slot_start < range_end && from < slot_end
        })
    }

    /// (un)plug a slot from an Hotpluggable memory region
    pub(crate) fn update_slot(
        &self,
        vm: &KvmVm,
        mem_slot: &GuestMemorySlot<'_>,
        plug: bool,
    ) -> Result<(), VmError> {
        // This function can only be called on hotpluggable regions!
        assert!(self.region_type == GuestRegionType::Hotpluggable);

        let mut bitmap_guard = self.plugged.lock().unwrap();
        let idx = (mem_slot.slot - self.slot_from) as usize;
        if bitmap_guard[idx] == plug {
            return Ok(());
        }

        // Commit the bitmap only once the protection and the KVM slot have both changed, rolling
        // back the first step if the second fails. A failed rollback has no way back, so it panics.
        let kvm_region = kvm_userspace_memory_region::from(mem_slot);
        if plug {
            // make it accessible _before_ adding it to KVM
            mem_slot.protect(false)?;
            if let Err(err) = vm.set_user_memory_region(kvm_region) {
                mem_slot
                    .protect(true)
                    .expect("cannot roll back the virtio-mem slot protection");
                return Err(err);
            }
            bitmap_guard.set(idx, true);
        } else {
            // to remove it we need to pass a size of zero
            let mut removed_region = kvm_region;
            removed_region.memory_size = 0;
            vm.set_user_memory_region(removed_region)?;
            // make it protected _after_ removing it from KVM
            if let Err(err) = mem_slot.protect(true) {
                vm.set_user_memory_region(kvm_region)
                    .expect("cannot roll back the virtio-mem slot in KVM");
                return Err(err.into());
            }
            bitmap_guard.set(idx, false);
        }
        Ok(())
    }

    pub(crate) fn discard_range(
        &self,
        caddr: MemoryRegionAddress,
        len: usize,
        partial_pages: PartialPages,
    ) -> Result<(), GuestMemoryError> {
        // The address and length can come straight from a guest descriptor (balloon), so reject
        // ranges that are not host page aligned or that run past the end of the region.
        let page_size = host_page_size() as u64;
        let start = caddr.raw_value();
        let aligned = start.is_multiple_of(page_size) && (len as u64).is_multiple_of(page_size);
        let Some(end) = start
            .checked_add(len as u64)
            .filter(|&end| aligned && end <= self.len())
        else {
            return Err(GuestMemoryError::InvalidGuestAddress(
                self.start_addr().unchecked_add(start),
            ));
        };

        // Only whole backing pages can be freed. With hugetlbfs, the partial huge pages at either
        // end of the range are zeroed instead, which allocates them if they were not backed yet,
        // unless `partial_pages` says to zero only where that allocates nothing. Linux guests
        // inflate the traditional balloon at most 1 MiB at a time, so with hugetlbfs, inflation
        // frees nothing, and with [`PartialPages::Zero`] it increases host memory use. Free page
        // hinting reports whole huge pages, and so does free page reporting by default.
        let backing_page_size = RawGuestRegionMmap::page_size(self.inner.flags()) as u64;
        let backing_aligned_start = align_up!(start, backing_page_size).min(end);
        let backing_aligned_end = align_down!(end, backing_page_size).max(backing_aligned_start);
        let backing_aligned_addr = MemoryRegionAddress(backing_aligned_start);
        let backing_aligned_len = u64_to_usize(backing_aligned_end - backing_aligned_start);
        let freed = match (self.inner.file_offset(), self.inner.flags()) {
            _ if backing_aligned_len == 0 => Ok(()),
            // If and only if we are resuming from a snapshot file, we have a file and it's mapped
            // private
            (Some(_), flags) if flags & libc::MAP_PRIVATE != 0 => {
                self.remap_anonymous_range(backing_aligned_addr, backing_aligned_len)
            }
            // MADV_REMOVE frees the backing pages of a shared file mapping and, unlike fallocate,
            // sends UFFD_EVENT_REMOVE to a registered userfaultfd.
            (Some(_), _) => {
                self.madvise_range(backing_aligned_addr, backing_aligned_len, libc::MADV_REMOVE)
            }
            (None, _) => self.madvise_range(
                backing_aligned_addr,
                backing_aligned_len,
                libc::MADV_DONTNEED,
            ),
        };
        // Freeing changes guest memory without a guest write, so KVM's dirty log never has it.
        // With dirty tracking on, the freed pages are recorded, and the next snapshot writes them
        // as zeros. A madvise that fails can have freed some pages of the range and not the rest,
        // so the range is marked written instead, and the next Diff reads it. It cannot read an
        // unplugged slot, so a failure in a range that holds one is remembered, and a Diff is
        // refused until a Full snapshot has been taken.
        let freed_start = u64_to_usize(backing_aligned_start);
        match freed {
            Ok(()) => self.discarded.mark_dirty(freed_start, backing_aligned_len),
            Err(_) => {
                (**self)
                    .bitmap()
                    .mark_dirty(freed_start, backing_aligned_len);
                if self
                    .check_range_plugged(backing_aligned_addr, backing_aligned_len)
                    .is_err()
                {
                    self.failed_unplugged_discard.store(true, Ordering::Relaxed);
                }
            }
        }
        freed?;

        for (edge_start, edge_end) in [(start, backing_aligned_start), (backing_aligned_end, end)] {
            let edge_addr = MemoryRegionAddress(edge_start);
            let edge_len = u64_to_usize(edge_end - edge_start);
            // Unplugged slots are PROT_NONE, and snapshots store them as holes.
            if edge_len == 0 || self.check_range_plugged(edge_addr, edge_len).is_err() {
                continue;
            }
            let edge = self.get_slice(edge_addr, edge_len)?;
            // An edge lies within one backing page, so its first host page answers for it. When
            // pagemap cannot be read the edge is zeroed.
            if partial_pages == PartialPages::ZeroWithoutAllocating
                && write_allocates(edge.ptr_guard().as_ptr() as usize, self.is_shared())
                    == Some(true)
            {
                continue;
            }
            // SAFETY: `get_slice` bounds `edge` within this region's mapping,
            // and plugged memory is writable.
            unsafe { std::ptr::write_bytes(edge.ptr_guard_mut().as_ptr(), 0, edge.len()) };
            // The dirty log does not have this write either.
            edge.bitmap().mark_dirty(0, edge.len());
        }
        Ok(())
    }

    /// Whether a discard failed in a range that holds an unplugged slot since the last
    /// [`Self::forget_failed_discard_of_unplugged_memory`].
    pub(crate) fn failed_discard_of_unplugged_memory(&self) -> bool {
        self.failed_unplugged_discard.load(Ordering::Relaxed)
    }

    /// Called once a Full snapshot has been written: later Diffs build on it.
    pub(crate) fn forget_failed_discard_of_unplugged_memory(&self) {
        self.failed_unplugged_discard
            .store(false, Ordering::Relaxed);
    }

    fn remap_anonymous_range(
        &self,
        caddr: MemoryRegionAddress,
        len: usize,
    ) -> Result<(), GuestMemoryError> {
        // Mmap a new anonymous region over the present one in order to create a hole
        // with zero pages.
        // This workaround is (only) needed after resuming from a snapshot file because the
        // guest memory is mmaped from file as private. In this case, MADV_DONTNEED on the
        // file only drops any anonymous pages in range, but subsequent accesses would read
        // whatever page is stored on the backing file. Mmapping anonymous pages ensures
        // it's zeroed.
        // Keep MAP_NORESERVE to match the flags the region was created with (see `create`).
        // Without it the replacement is subject to overcommit accounting and can fail with
        // ENOMEM, by which point MAP_FIXED has already unmapped the range.
        // TODO: this does not re-apply the region's madvise flags, so it would drop the
        // MADV_HUGEPAGE hint on a THP + file-restore VM. That combination is unsupported
        // today; revisit if it becomes supported.
        //
        // Map each intersecting slot directly to its final protection: fully unplugged
        // slots get PROT_NONE, plugged (or mixed) slots stay PROT_READ | PROT_WRITE. This
        // avoids mapping the whole range PROT_READ | PROT_WRITE and then reprotecting the
        // unplugged slots.
        let base = self.start_addr().raw_value();
        let from = base
            .checked_add(caddr.raw_value())
            .expect("caddr should be within the region");
        let to = from
            .checked_add(len as u64)
            .expect("range should be within the region");
        for (slot, plugged) in self.slots_intersecting_range(GuestAddress(from), len) {
            let slot_start = slot.guest_addr.raw_value();
            let slot_end = slot_start
                .checked_add(slot.slice.len() as u64)
                .expect("slot should be within the region");
            let start = slot_start.max(from);
            let end = slot_end.min(to);
            let host = self.get_host_address(MemoryRegionAddress(start - base))?;
            let prot = if plugged {
                libc::PROT_READ | libc::PROT_WRITE
            } else {
                libc::PROT_NONE
            };
            // SAFETY: [start, end) is clamped to this slot, so `host` and the length are
            // within the region's existing mapping. MAP_FIXED replaces exactly that range.
            let ret = unsafe {
                libc::mmap(
                    host.cast(),
                    u64_to_usize(end - start),
                    prot,
                    libc::MAP_FIXED | libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            // MAP_FIXED has already unmapped the old range, so a failure here cannot be
            // recovered from.
            if ret == libc::MAP_FAILED {
                panic!(
                    "discard_range: mmap failed: {:?}",
                    std::io::Error::last_os_error()
                );
            }
        }
        Ok(())
    }

    fn madvise_range(
        &self,
        caddr: MemoryRegionAddress,
        len: usize,
        advice: libc::c_int,
    ) -> Result<(), GuestMemoryError> {
        let host_addr = self.get_host_address(caddr)?;
        // SAFETY: `discard_range` checked that the range is aligned and lies within this region.
        let ret = unsafe { libc::madvise(host_addr.cast(), len, advice) };
        if ret < 0 {
            let os_error = std::io::Error::last_os_error();
            error!("discard_range: madvise failed: {:?}", os_error);
            return Err(GuestMemoryError::IOError(os_error));
        }
        Ok(())
    }
}

impl Deref for GuestRegionMmapExt {
    type Target = MmapRegion<Option<AtomicBitmap>>;

    fn deref(&self) -> &MmapRegion<Option<AtomicBitmap>> {
        &self.inner
    }
}

impl GuestMemoryRegionBytes for GuestRegionMmapExt {}

#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
impl GuestMemoryRegion for GuestRegionMmapExt {
    type B = Option<AtomicBitmap>;

    fn len(&self) -> GuestUsize {
        self.inner.len()
    }

    fn start_addr(&self) -> GuestAddress {
        self.inner.start_addr()
    }

    fn bitmap(&self) -> BS<'_, Self::B> {
        self.inner.bitmap()
    }

    fn get_host_address(
        &self,
        addr: MemoryRegionAddress,
    ) -> vm_memory::guest_memory::Result<*mut u8> {
        self.inner.get_host_address(addr)
    }

    fn file_offset(&self) -> Option<&FileOffset> {
        self.inner.file_offset()
    }

    fn get_slice(
        &self,
        offset: MemoryRegionAddress,
        count: usize,
    ) -> vm_memory::guest_memory::Result<VolatileSlice<'_, BS<'_, Self::B>>> {
        self.inner.get_slice(offset, count)
    }
}

/// The filesystem holding a guest memory backing file, as reported by `fstatfs(2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackingFileSystem {
    // The `statfs` field types differ between libc targets, so both are widened to i128.
    magic: i128,
    block_size: i128,
}

impl BackingFileSystem {
    /// Reads the filesystem type and block size of `file`.
    pub fn of(file: &File) -> io::Result<Self> {
        let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: `file` owns a valid descriptor and `stat` is writable storage for one `statfs`.
        if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a successful `fstatfs` initialized `stat`.
        let stat = unsafe { stat.assume_init() };
        Ok(Self {
            magic: i128::from(stat.f_type),
            block_size: i128::from(stat.f_bsize),
        })
    }

    /// Returns the huge page size in bytes when the file is on hugetlbfs, whose block size is its
    /// page size, and `None` for any other filesystem.
    pub fn hugetlbfs_page_size(&self) -> Option<u64> {
        if self.magic == i128::from(libc::HUGETLBFS_MAGIC) {
            u64::try_from(self.block_size).ok().filter(|size| *size > 0)
        } else {
            None
        }
    }

    /// Returns whether the file is on tmpfs, where an ordinary memfd lives.
    pub fn is_shmem(&self) -> bool {
        self.magic == i128::from(libc::TMPFS_MAGIC)
    }
}

impl std::fmt::Display for BackingFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(page_size) = self.hugetlbfs_page_size() {
            write!(f, "hugetlbfs with {page_size}-byte pages")
        } else if self.is_shmem() {
            write!(f, "shmem")
        } else {
            write!(f, "a filesystem with magic {:#x}", self.magic)
        }
    }
}

/// Creates a `Vec` of anonymous `GuestRegionMmap` with the given configuration.
pub fn memory_regions_from_ranges(
    regions: &[(GuestAddress, usize)],
    mmap_flags: libc::c_int,
    track_dirty_pages: bool,
    madvise_flags: libc::c_int,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    regions
        .iter()
        .map(|&(start, size)| {
            memory_region_from_range(
                start,
                size,
                libc::MAP_NORESERVE | mmap_flags,
                None,
                track_dirty_pages,
                madvise_flags,
            )
        })
        .collect()
}

/// Creates a `Vec` of `GuestRegionMmap` mapped from a file: the first region at `file_offset`,
/// each following one right after the previous.
pub fn memory_regions_from_ranges_file_backed(
    regions: &[(GuestAddress, usize)],
    mmap_flags: libc::c_int,
    file_offset: FileOffset,
    track_dirty_pages: bool,
    madvise_flags: libc::c_int,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    map_file_ranges(
        regions,
        libc::MAP_NORESERVE | mmap_flags,
        file_offset,
        track_dirty_pages,
        madvise_flags,
    )
}

/// [`memory_regions_from_ranges_file_backed`] with `mmap_flags` passed to `mmap(2)` as they
/// are, so the caller decides whether the mapping is `MAP_NORESERVE`.
fn map_file_ranges(
    regions: &[(GuestAddress, usize)],
    mmap_flags: libc::c_int,
    file_offset: FileOffset,
    track_dirty_pages: bool,
    madvise_flags: libc::c_int,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    let end = regions
        .iter()
        .try_fold(file_offset.start(), |acc, &(_, size)| {
            acc.checked_add(size as u64)
        })
        .ok_or(MemoryError::OffsetTooLarge)?;
    // Use the end offset rather than `fstat`: a block device reports `st_size == 0`.
    let file_size = file_offset
        .file()
        .seek(SeekFrom::End(0))
        .map_err(MemoryError::SeekError)?;

    // Ensure we do not mmap beyond EOF. The kernel would allow that, but a SIGBUS is triggered
    // on an attempted access to a page of the mapping that lies beyond the end of the file.
    if end > file_size {
        return Err(MemoryError::OffsetTooLarge);
    }

    let mut offset = file_offset.start();
    regions
        .iter()
        .map(|&(start, size)| {
            let guest_memory = memory_region_from_range(
                start,
                size,
                mmap_flags,
                Some(FileOffset::from_arc(Arc::clone(file_offset.arc()), offset)),
                track_dirty_pages,
                madvise_flags,
            )?;
            offset += size as u64;
            Ok(guest_memory)
        })
        .collect()
}

/// Creates a single `GuestRegionMmap`, mapped from `file_offset` if given. `mmap_flags` are
/// passed to `mmap(2)` as they are.
fn memory_region_from_range(
    start: GuestAddress,
    size: usize,
    mmap_flags: libc::c_int,
    file_offset: Option<FileOffset>,
    track_dirty_pages: bool,
    madvise_flags: libc::c_int,
) -> Result<GuestRegionMmap, MemoryError> {
    let guest_memory = GuestRegionMmap::allocate(
        start,
        size,
        libc::PROT_READ | libc::PROT_WRITE,
        mmap_flags,
        file_offset,
        track_dirty_pages,
    )?;
    // SAFETY: The referenced memory was just mapped.
    let ret = unsafe {
        libc::madvise(
            guest_memory.as_ptr().cast(),
            guest_memory.size(),
            madvise_flags,
        )
    };
    if ret != 0 {
        return Err(MemoryError::Madvise(io::Error::last_os_error()));
    }
    Ok(guest_memory)
}

/// Creates a GuestMemoryMmap from raw regions.
pub fn anonymous(
    regions: &[(GuestAddress, usize)],
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    memory_regions_from_ranges(
        regions,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | huge_pages.mmap_flags(),
        track_dirty_pages,
        huge_pages.madvise_flags(),
    )
}

/// A single memfd backing all guest memory regions, laid out like a guest memory snapshot file.
///
/// Regions are allocated in order with [`MemfdBacking::allocate`]. The memfd is sized up front
/// and sealed, so the total size of all regions must be known when creating it.
#[derive(Debug)]
pub struct MemfdBacking {
    /// The memfd. In an [Arc], so it can be placed inside a [FileOffset].
    pub file: Arc<File>,
    /// Size of the part of the memfd that regions have been mapped from so far. The next region
    /// is mapped at this offset.
    allocated_size: u64,
    /// Guest address at which the last allocated region ends. The next region must start at or
    /// after it.
    next_guest_addr: GuestAddress,
}

impl MemfdBacking {
    /// Creates a sealed memfd of `total_size` bytes.
    pub fn new(total_size: u64, huge_pages: HugePageConfig) -> Result<Self, MemoryError> {
        let memfd_file = create_memfd(total_size, huge_pages.into())?.into_file();
        Ok(Self {
            file: Arc::new(memfd_file),
            allocated_size: 0,
            next_guest_addr: GuestAddress(0),
        })
    }

    /// Maps `regions` `MAP_SHARED` from the memfd, one after the other, right after the regions
    /// allocated so far. Fails if they do not fit in what is left of the memfd.
    ///
    /// # Panics
    ///
    /// If `regions` are not in increasing guest address order, or if any of them starts below the
    /// end of a previously allocated region.
    pub fn allocate(
        &mut self,
        regions: &[(GuestAddress, usize)],
        track_dirty_pages: bool,
        huge_pages: HugePageConfig,
    ) -> Result<Vec<GuestRegionMmap>, MemoryError> {
        let mut next_guest_addr = self.next_guest_addr;
        for &(start, size) in regions {
            assert!(
                start >= next_guest_addr,
                "memfd-backed region at {:#x} allocated after a region ending at {:#x}:\
                 regions should be allocated in increasing order.",
                start.0,
                next_guest_addr.0
            );
            next_guest_addr = start
                .checked_add(size as u64)
                .ok_or(MemoryError::OffsetTooLarge)?;
        }

        let guest_memory = memory_regions_from_ranges_file_backed(
            regions,
            libc::MAP_SHARED | huge_pages.mmap_flags(),
            FileOffset::from_arc(Arc::clone(&self.file), self.allocated_size),
            track_dirty_pages,
            huge_pages.madvise_flags(),
        )?;
        // `memory_regions_from_ranges_file_backed` checked that every region ends within the
        // memfd.
        self.allocated_size += guest_memory.iter().map(|r| r.len()).sum::<u64>();
        self.next_guest_addr = next_guest_addr;
        Ok(guest_memory)
    }
}

/// Creates a GuestMemoryMmap given a `file` containing the data
/// and a `state` containing mapping information.
pub fn snapshot_file(
    file: File,
    regions: &[(GuestAddress, usize)],
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    memory_regions_from_ranges_file_backed(
        regions,
        libc::MAP_PRIVATE,
        FileOffset::new(file, 0),
        track_dirty_pages,
        huge_pages.madvise_flags(),
    )
}

/// Maps the backing file of a UFFD minor-fault handler, which holds the guest memory of a
/// snapshot laid out like its memory file: every region `MAP_PRIVATE`, one after the other
/// from offset 0. A clean page stays shared with every other mapping of the file, and a guest
/// write copies it.
///
/// A mapping of shmem is `MAP_NORESERVE`, like every other guest memory mapping: its copies
/// are ordinary anonymous memory. A mapping of hugetlbfs is not, so the kernel reserves a
/// huge page for the copy of each one it maps: a pool too small for that fails this call with
/// `ENOMEM`, where an unreserved mapping would deliver `SIGBUS` to a running guest at the
/// write the pool cannot serve.
pub fn uffd_minor_backing(
    file: File,
    regions: &[(GuestAddress, usize)],
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<Vec<GuestRegionMmap>, MemoryError> {
    let reservation = if huge_pages.is_hugetlbfs() {
        0
    } else {
        libc::MAP_NORESERVE
    };
    map_file_ranges(
        regions,
        libc::MAP_PRIVATE | reservation | huge_pages.mmap_flags(),
        FileOffset::new(file, 0),
        track_dirty_pages,
        huge_pages.madvise_flags(),
    )
}

/// The bytes of guest memory in `regions` that the file byte ranges `ranges` map, each range
/// clamped to the regions.
pub fn file_ranges_len(regions: &[GuestRegionMmap], ranges: &[(u64, u64)]) -> u64 {
    let bounds: Vec<(u64, u64)> = regions
        .iter()
        .filter_map(|region| {
            let start = region.file_offset()?.start();
            Some((start, start + region.len()))
        })
        .collect();
    clamped_len(&bounds, ranges)
}

/// The summed length of `ranges` clamped to each of `bounds` (file offset start and end).
/// It saturates: overlapping ranges must not wrap past `u64::MAX` to a small number, which
/// would pass a bound on the total.
fn clamped_len(bounds: &[(u64, u64)], ranges: &[(u64, u64)]) -> u64 {
    let mut total = 0u64;
    for &(start, end) in bounds {
        for &(offset, length) in ranges {
            let from = offset.max(start);
            let to = offset.saturating_add(length).min(end);
            total = total.saturating_add(to.saturating_sub(from));
        }
    }
    total
}

/// Populates writable, with MADV_POPULATE_WRITE, the guest memory that maps the byte ranges
/// `ranges` (offset, length) of the file backing `regions`: on a private file mapping each page
/// is copied from the page cache into this process now, rather than on the guest's first write
/// under a stage-2 fault. The parts of a range outside every region are skipped. Returns the
/// bytes populated.
pub fn populate_write_file_ranges(
    regions: &[GuestRegionMmap],
    ranges: &[(u64, u64)],
) -> Result<u64, MemoryError> {
    let mut populated = 0u64;
    for region in regions {
        let Some(file_offset) = region.file_offset() else {
            continue;
        };
        let start = file_offset.start();
        let end = start + region.len();
        for &(offset, length) in ranges {
            let from = offset.max(start);
            let to = offset.saturating_add(length).min(end);
            if from >= to {
                continue;
            }
            // SAFETY: [from, to) lies inside this region's mapping, which `regions` keeps mapped.
            let ret = unsafe {
                libc::madvise(
                    region
                        .as_ptr()
                        .add(usize::try_from(from - start).unwrap())
                        .cast(),
                    usize::try_from(to - from).unwrap(),
                    libc::MADV_POPULATE_WRITE,
                )
            };
            if ret != 0 {
                return Err(MemoryError::WriteMemory(GuestMemoryError::IOError(
                    io::Error::last_os_error(),
                )));
            }
            populated += to - from;
        }
    }
    Ok(populated)
}

/// Marks as dirty the `len` bytes of guest memory mapped at host address `ptr`.
///
/// `[ptr, ptr + len)` must lie within a single region of `mem`, as ranges resolved with
/// `get_slice` do.
///
/// # Panics
///
/// Panics if `len > 0` and the range does not fall within a single region.
pub fn mark_dirty_host(mem: &GuestMemoryMmap, ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    let start = ptr as usize;
    let end = start.strict_add(len);
    for region in mem.iter() {
        let region_start = region.as_ptr() as usize;
        let region_end = region_start.strict_add(region.size());
        if region_start <= start && end <= region_end {
            region
                .bitmap()
                .mark_dirty(start.strict_sub(region_start), len);
            return;
        }
    }
    unreachable!("host range {ptr:p}+{len:#x} is not within a single guest memory region");
}

/// Marks dirty in `bitmap` the pages whose bits are set in `bits`.
fn set_bits(bitmap: &AtomicBitmap, bits: &[u64]) {
    for (i, &word) in bits.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            bitmap.set_bit(i * 64 + word.trailing_zeros() as usize);
            word &= word - 1;
        }
    }
}

/// Defines the interface for snapshotting memory.
pub trait GuestMemoryExtension
where
    Self: Sized,
{
    /// Describes GuestMemoryMmap through a GuestMemoryState struct.
    fn describe(&self) -> GuestMemoryState;

    /// Mark memory range as dirty
    fn mark_dirty(&self, addr: GuestAddress, len: usize);

    /// Dumps all contents of GuestMemoryMmap to a writer. With [`ZeroPages::Hole`], all-zero
    /// pages are seeked over instead of written, and with [`AbsentPages::Skip`] so are the pages
    /// of anonymous memory that have no page table entry, without reading them. An unplugged slot
    /// cannot be read: its pages in `discarded` ([`Self::take_discarded`]) are written as zeros,
    /// or left as holes with [`ZeroPages::Hole`], and the rest of it is seeked over, so the file
    /// keeps what it held there.
    fn dump<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        zero_pages: ZeroPages,
        discarded: &DirtyBitmap,
    ) -> Result<(), MemoryError>;

    /// Dumps all pages of GuestMemoryMmap present in `dirty_bitmap` or `discarded`
    /// ([`Self::take_discarded`]) to a writer and seeks over the rest. A page in `discarded` is
    /// written as zeros, an unplugged slot's included, and read only when a plugged slot's page
    /// was also written and may hold data.
    fn dump_dirty<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        dirty_bitmap: &DirtyBitmap,
        discarded: &DirtyBitmap,
    ) -> Result<(), MemoryError>;

    /// Takes Firecracker's dirty bits, one bitmap per region, clearing them. Every region must
    /// track dirty pages.
    #[must_use]
    fn take_dirty(&self) -> DirtyBitmap;

    /// Takes the pages a discard emptied since the last take, one bitmap per region, clearing
    /// them. A region that does not track dirty pages records none.
    #[must_use]
    fn take_discarded(&self) -> DirtyBitmap;

    /// Resets all the memory region bitmaps
    #[cfg(test)]
    fn reset_dirty(&self);

    /// ORs into `dirty_bitmap` each plugged slot's log, as `read_log` returns it. Stops at the
    /// first log that fails to read.
    fn merge_slot_kvm_dirty_logs<E, F>(
        &self,
        dirty_bitmap: &mut DirtyBitmap,
        read_log: F,
    ) -> Result<(), E>
    where
        F: FnMut(&GuestMemorySlot<'_>) -> Result<Vec<u64>, E>;

    /// Marks dirty again every page set in `dirty_bitmap`.
    fn store_dirty_bitmap(&self, dirty_bitmap: &DirtyBitmap);

    /// Marks discarded again every page set in `discarded`.
    fn store_discarded(&self, discarded: &DirtyBitmap);

    /// Apply a function to each region in a memory range
    fn try_for_each_region_in_range<F>(
        &self,
        addr: GuestAddress,
        range_len: usize,
        f: F,
    ) -> Result<(), GuestMemoryError>
    where
        F: FnMut(&GuestRegionMmapExt, MemoryRegionAddress, usize) -> Result<(), GuestMemoryError>;

    /// Discards a memory range, freeing the whole backing pages inside it and zeroing the rest.
    /// With dirty tracking on, the pages it frees are recorded for [`Self::take_discarded`] and
    /// the pages it zeroes are marked dirty. When freeing fails the range is marked dirty
    /// instead.
    fn discard_range(&self, addr: GuestAddress, range_len: usize) -> Result<(), GuestMemoryError>;

    /// Discards a memory range a guest gave to the traditional balloon: as
    /// [`Self::discard_range`], except that a huge page the range covers part of is zeroed
    /// only where that allocates nothing ([`PartialPages::ZeroWithoutAllocating`]). A guest
    /// may not rely on what a page in the balloon holds, and it gets the page back with the
    /// bytes this VM had there before, so nothing but the cost differs: zeroing an unbacked
    /// huge page for 4 KiB of balloon allocates 2 MiB, and on memory mapped privately from a
    /// file shared between VMs it turns a shared page into a private one.
    fn discard_ballooned_range(
        &self,
        addr: GuestAddress,
        range_len: usize,
    ) -> Result<(), GuestMemoryError>;

    /// Check whether the given guest address range falls entirely within plugged memory.
    /// Returns Err if the address is not in any region or is in an unplugged slot.
    fn check_range_plugged(&self, addr: GuestAddress, len: usize) -> Result<(), GuestMemoryError>;
}

/// State of a guest memory region saved to file/buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMemoryRegionState {
    // This should have been named `base_guest_addr` since it's _guest_ addr, but for
    // backward compatibility we have to keep this name. At least this comment should help.
    /// Base GuestAddress.
    pub base_address: u64,
    /// Region size.
    pub size: usize,
    /// Region type
    pub region_type: GuestRegionType,
    /// Plugged/unplugged status of each slot
    pub plugged: Vec<bool>,
}

/// Describes guest memory regions and their snapshot file mappings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMemoryState {
    /// List of regions.
    pub regions: Vec<GuestMemoryRegionState>,
}

impl GuestMemoryState {
    /// Turns this [`GuestMemoryState`] into a description of guest memory regions as understood
    /// by the creation functions of [`GuestMemoryExtensions`]
    pub fn regions(&self) -> Vec<(GuestAddress, usize)> {
        self.regions
            .iter()
            .map(|region| (GuestAddress(region.base_address), region.size))
            .collect()
    }
}

impl GuestMemoryExtension for GuestMemoryMmap {
    /// Describes GuestMemoryMmap through a GuestMemoryState struct.
    fn describe(&self) -> GuestMemoryState {
        let mut guest_memory_state = GuestMemoryState::default();
        self.iter().for_each(|region| {
            guest_memory_state.regions.push(GuestMemoryRegionState {
                base_address: region.start_addr().0,
                size: u64_to_usize(region.len()),
                region_type: region.region_type,
                plugged: region.plugged.lock().unwrap().iter().by_vals().collect(),
            });
        });
        guest_memory_state
    }

    /// Mark memory range as dirty
    fn mark_dirty(&self, addr: GuestAddress, len: usize) {
        // ignore invalid ranges using .flatten()
        for slice in self.get_slices(addr, len).flatten() {
            slice.bitmap().mark_dirty(0, slice.len());
        }
    }

    /// Dumps all contents of GuestMemoryMmap to a writer. With [`ZeroPages::Hole`], all-zero
    /// pages are seeked over instead of written, and with [`AbsentPages::Skip`] so are the pages
    /// of anonymous memory that have no page table entry, without reading them. An unplugged slot
    /// cannot be read: its pages in `discarded` ([`Self::take_discarded`]) are written as zeros,
    /// or left as holes with [`ZeroPages::Hole`], and the rest of it is seeked over, so the file
    /// keeps what it held there.
    fn dump<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        zero_pages: ZeroPages,
        discarded: &DirtyBitmap,
    ) -> Result<(), MemoryError> {
        let page_size = host_page_size();
        let mut scratch = DumpScratch::default();

        assert_eq!(discarded.len(), self.num_regions());
        for (region, region_discarded) in self.iter().zip(discarded) {
            // A page of a file mapping that has no page table entry holds the file's data.
            let anonymous = region.file_offset().is_none();
            for (j, (mem_slot, plugged)) in region.slots().enumerate() {
                match (zero_pages, plugged) {
                    (ZeroPages::Write, true) => write_memory(writer, &mem_slot.slice)?,
                    (ZeroPages::Write, false) => mem_slot.dump_dirty(
                        writer,
                        None,
                        &region_discarded[region.slot_words(j)],
                        page_size,
                        &mut scratch,
                    )?,
                    (ZeroPages::Hole(absent_pages), true) => {
                        let skip_absent = anonymous && absent_pages == AbsentPages::Skip;
                        mem_slot.dump_nonzero(writer, skip_absent.then_some(&mut scratch))?
                    }
                    // The file holds no data, so it already reads as zeros under the slot.
                    (ZeroPages::Hole(_), false) => seek_forward(writer, mem_slot.slice.len())?,
                }
            }
        }
        Ok(())
    }

    /// Dumps all pages of GuestMemoryMmap present in `dirty_bitmap` or `discarded`
    /// ([`Self::take_discarded`]) to a writer and seeks over the rest. A page in `discarded` is
    /// written as zeros, an unplugged slot's included, and read only when a plugged slot's page
    /// was also written and may hold data.
    fn dump_dirty<T: WriteVolatile + std::io::Seek>(
        &self,
        writer: &mut T,
        dirty_bitmap: &DirtyBitmap,
        discarded: &DirtyBitmap,
    ) -> Result<(), MemoryError> {
        let page_size = host_page_size();
        let mut scratch = DumpScratch::default();

        assert_eq!(dirty_bitmap.len(), self.num_regions());
        assert_eq!(discarded.len(), self.num_regions());
        for ((region, region_bitmap), region_discarded) in
            self.iter().zip(dirty_bitmap).zip(discarded)
        {
            assert_eq!(
                region_bitmap.len(),
                (region.size() / page_size).div_ceil(64)
            );
            for (j, (mem_slot, plugged)) in region.slots().enumerate() {
                let words = region.slot_words(j);
                mem_slot.dump_dirty(
                    writer,
                    plugged.then(|| &region_bitmap[words.clone()]),
                    &region_discarded[words],
                    page_size,
                    &mut scratch,
                )?;
            }
        }
        Ok(())
    }

    /// Takes Firecracker's dirty bits, one bitmap per region, clearing them. Every region must
    /// track dirty pages.
    fn take_dirty(&self) -> DirtyBitmap {
        self.iter()
            .map(|region| {
                (**region)
                    .bitmap()
                    .as_ref()
                    .expect("region should track dirty pages")
                    .get_and_reset()
            })
            .collect()
    }

    /// Takes the pages a discard emptied since the last take, one bitmap per region, clearing
    /// them. A region that does not track dirty pages records none.
    fn take_discarded(&self) -> DirtyBitmap {
        self.iter()
            .map(|region| match &region.discarded {
                Some(discarded) => discarded.get_and_reset(),
                None => vec![0; (region.size() / host_page_size()).div_ceil(64)],
            })
            .collect()
    }

    /// Resets all the memory region bitmaps
    #[cfg(test)]
    fn reset_dirty(&self) {
        self.iter().for_each(|region| {
            if let Some(bitmap) = (**region).bitmap() {
                bitmap.reset();
            }
        })
    }

    /// ORs into `dirty_bitmap` each plugged slot's log, as `read_log` returns it. Stops at the
    /// first log that fails to read.
    fn merge_slot_kvm_dirty_logs<E, F>(
        &self,
        dirty_bitmap: &mut DirtyBitmap,
        mut read_log: F,
    ) -> Result<(), E>
    where
        F: FnMut(&GuestMemorySlot<'_>) -> Result<Vec<u64>, E>,
    {
        for (region, region_bitmap) in self.iter().zip(dirty_bitmap.iter_mut()) {
            for (i, (mem_slot, plugged)) in region.slots().enumerate() {
                if plugged {
                    let log = read_log(&mem_slot)?;
                    let words = &mut region_bitmap[region.slot_words(i)];
                    assert_eq!(log.len(), words.len());
                    for (word, bits) in words.iter_mut().zip(log) {
                        *word |= bits;
                    }
                }
            }
        }
        Ok(())
    }

    /// Marks dirty again every page set in `dirty_bitmap`.
    fn store_dirty_bitmap(&self, dirty_bitmap: &DirtyBitmap) {
        for (region, region_bitmap) in self.iter().zip(dirty_bitmap) {
            if let Some(bitmap) = (**region).bitmap() {
                set_bits(bitmap, region_bitmap);
            }
        }
    }

    /// Marks discarded again every page set in `discarded`.
    fn store_discarded(&self, discarded: &DirtyBitmap) {
        for (region, region_discarded) in self.iter().zip(discarded) {
            if let Some(bitmap) = &region.discarded {
                set_bits(bitmap, region_discarded);
            }
        }
    }

    fn try_for_each_region_in_range<F>(
        &self,
        addr: GuestAddress,
        range_len: usize,
        mut f: F,
    ) -> Result<(), GuestMemoryError>
    where
        F: FnMut(&GuestRegionMmapExt, MemoryRegionAddress, usize) -> Result<(), GuestMemoryError>,
    {
        let mut cur = addr;
        let mut remaining = range_len;

        // iterate over all adjacent consecutive regions in range
        while let Some(region) = self.find_region(cur) {
            let start = region.to_region_addr(cur).unwrap();
            let len = std::cmp::min(
                // remaining bytes inside the region
                u64_to_usize(region.len() - start.raw_value()),
                // remaning bytes to discard
                remaining,
            );

            f(region, start, len)?;

            remaining -= len;
            if remaining == 0 {
                return Ok(());
            }

            cur = cur
                .checked_add(len as u64)
                .ok_or(GuestMemoryError::GuestAddressOverflow)?;
        }
        // if we exit the loop because we didn't find a region, return an error
        Err(GuestMemoryError::InvalidGuestAddress(cur))
    }

    fn discard_range(&self, addr: GuestAddress, range_len: usize) -> Result<(), GuestMemoryError> {
        self.try_for_each_region_in_range(addr, range_len, |region, start, len| {
            region.discard_range(start, len, PartialPages::Zero)
        })
    }

    fn discard_ballooned_range(
        &self,
        addr: GuestAddress,
        range_len: usize,
    ) -> Result<(), GuestMemoryError> {
        self.try_for_each_region_in_range(addr, range_len, |region, start, len| {
            region.discard_range(start, len, PartialPages::ZeroWithoutAllocating)
        })
    }

    fn check_range_plugged(&self, addr: GuestAddress, len: usize) -> Result<(), GuestMemoryError> {
        self.try_for_each_region_in_range(addr, len, |region, offset, chunk_len| {
            region.check_range_plugged(offset, chunk_len)
        })
    }
}

fn create_memfd(
    mem_size: u64,
    hugetlb_size: Option<memfd::HugetlbSize>,
) -> Result<memfd::Memfd, MemoryError> {
    // Create a memfd.
    let opts = memfd::MemfdOptions::default()
        .hugetlb(hugetlb_size)
        .allow_sealing(true);
    let mem_file = opts.create("guest_mem").map_err(MemoryError::Memfd)?;

    // Resize to guest mem size.
    mem_file
        .as_file()
        .set_len(mem_size)
        .map_err(MemoryError::MemfdSetLen)?;

    // Add seals to prevent further resizing.
    let mut seals = memfd::SealsHashSet::new();
    seals.insert(memfd::FileSeal::SealShrink);
    seals.insert(memfd::FileSeal::SealGrow);
    mem_file.add_seals(&seals).map_err(MemoryError::Memfd)?;

    // Prevent further sealing changes.
    mem_file
        .add_seal(memfd::FileSeal::SealSeal)
        .map_err(MemoryError::Memfd)?;

    Ok(mem_file)
}

/// Test utilities
pub mod test_utils {
    use super::*;

    /// Returns the data runs of `file` as (offset, len) pairs, found with SEEK_DATA and
    /// SEEK_HOLE, so a test sees which ranges the filesystem holds as holes.
    #[cfg(test)]
    pub(crate) fn data_runs(file: &File) -> Vec<(u64, u64)> {
        let fd = file.as_raw_fd();
        let len = i64::try_from(file.metadata().unwrap().len()).unwrap();
        let mut runs = Vec::new();
        let mut pos = 0;
        while pos < len {
            // SAFETY: lseek on an open descriptor touches no memory.
            let data = unsafe { libc::lseek(fd, pos, libc::SEEK_DATA) };
            if data < 0 {
                let err = io::Error::last_os_error();
                // ENXIO: no data at or after `pos`.
                assert_eq!(
                    err.raw_os_error(),
                    Some(libc::ENXIO),
                    "SEEK_DATA at {pos}: {err}"
                );
                break;
            }
            // SAFETY: lseek on an open descriptor touches no memory.
            let hole = unsafe { libc::lseek(fd, data, libc::SEEK_HOLE) };
            assert!(
                hole > data,
                "SEEK_HOLE at {data}: {}",
                io::Error::last_os_error()
            );
            runs.push((
                u64::try_from(data).unwrap(),
                u64::try_from(hole - data).unwrap(),
            ));
            pos = hole;
        }
        runs
    }

    /// Converts a vec of GuestRegionMmap into a GuestMemoryMmap using GuestRegionMmapExt
    pub fn into_region_ext(regions: Vec<GuestRegionMmap>) -> GuestMemoryMmap {
        GuestMemoryMmap::from_regions(
            regions
                .into_iter()
                .zip(0u32..) // assign dummy slots
                .map(|(region, slot)| GuestRegionMmapExt::dram_from_mmap_region(region, slot))
                .collect(),
        )
        .unwrap()
    }

    /// Creates a memfd of `len` bytes filled with `fill`: on hugetlbfs with 2 MiB pages when
    /// `hugetlbfs` is set, on shmem otherwise. hugetlbfs has no `write(2)`, so the contents are
    /// written through a shared mapping, which also claims the huge pages up front.
    ///
    /// Panics, naming the host requirement, when the huge page pool cannot back the file.
    #[cfg(test)]
    pub(crate) fn test_memfd(len: usize, hugetlbfs: bool, fill: u8) -> File {
        use std::os::fd::FromRawFd;

        let mut flags = libc::MFD_CLOEXEC;
        if hugetlbfs {
            flags |= libc::MFD_HUGETLB | libc::MFD_HUGE_2MB;
        }
        // SAFETY: the name is a valid NUL-terminated string.
        let fd = unsafe { libc::memfd_create(c"firecracker_test_memfd".as_ptr(), flags) };
        assert!(
            fd >= 0,
            "memfd_create failed: {}",
            io::Error::last_os_error()
        );
        // SAFETY: `fd` was just created and has no other owner.
        let file = unsafe { File::from_raw_fd(fd) };
        file.set_len(len as u64).unwrap();
        // SAFETY: a new shared mapping of the first `len` bytes of `file`, unmapped below.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(
            addr,
            libc::MAP_FAILED,
            "mapping the test memfd failed: {}{}",
            io::Error::last_os_error(),
            if hugetlbfs {
                "; this test needs free 2 MiB huge pages (vm.nr_hugepages)"
            } else {
                ""
            }
        );
        // SAFETY: `addr` maps `len` writable bytes that nothing else references.
        unsafe {
            std::ptr::write_bytes(addr.cast::<u8>(), fill, len);
            libc::munmap(addr, len);
        }
        file
    }

    /// Serializes the unit tests that allocate 2 MiB huge pages and checks the pool first, so a
    /// host needs only the pages of the largest single test, whatever the harness runs at once.
    ///
    /// Like `test_guest_region_alignment_and_size_huge`, a test skips when the host cannot
    /// provide its pages: this prints why and returns `None`.
    #[cfg(test)]
    pub(crate) fn lock_free_huge_pages(pages: u64) -> Option<std::sync::MutexGuard<'static, ()>> {
        static HUGE_PAGE_TESTS: Mutex<()> = Mutex::new(());
        let guard = HUGE_PAGE_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let counter = |name: &str| {
            std::fs::read_to_string(format!("/sys/kernel/mm/hugepages/hugepages-2048kB/{name}"))
                .ok()?
                .trim()
                .parse::<u64>()
                .ok()
        };
        // Pages that another mapping has reserved count as free but cannot be claimed.
        let available = counter("free_hugepages")
            .zip(counter("resv_hugepages"))
            .map(|(free, reserved)| free.saturating_sub(reserved));
        if available.is_some_and(|available| available >= pages) {
            return Some(guard);
        }
        let available = available.map_or_else(|| "none".to_string(), |count| count.to_string());
        println!(
            "Skipping: this test needs {pages} free 2 MiB hugepages and {available} are \
             available; raise vm.nr_hugepages to provide them"
        );
        None
    }
}

/// Seconds a new guest's memory stays out of automatic NUMA balancing; 0 leaves it in from the
/// start. Set once from `--numa-balancing-after-secs` before any guest memory exists.
pub static NUMA_BALANCING_AFTER_SECS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `mbind(2)` one guest memory range to `mode` with no node mask.
fn mbind_range(addr: usize, len: usize, mode: libc::c_int) -> io::Result<()> {
    // SAFETY: mbind changes only the memory policy of the range, which this process maps for
    // the guest's lifetime; a null node mask is valid for MPOL_LOCAL and MPOL_DEFAULT.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr,
            len,
            mode,
            std::ptr::null::<libc::c_ulong>(),
            0 as libc::c_ulong,
            0 as libc::c_uint,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Keep `memory` out of automatic NUMA balancing for [`NUMA_BALANCING_AFTER_SECS`], then let
/// it in.
///
/// A balancing scan unmaps the memory it walks from stage 2, and the guest faults back every
/// page it touches next; a restored clone's anonymous pages are also hinted and migrated. In
/// the seconds a clone takes to start, that is most of its cost on a two-node host. A guest
/// that lives on gains from balancing, so the exclusion ends: the range is bound MPOL_LOCAL
/// (which allocates where the default policy does, but has no MPOL_F_MOF, so
/// `task_numa_work` skips the VMA), and a thread of its own returns it to MPOL_DEFAULT when
/// the time is up. The thread runs under `filter` (the VMM thread's) and waits on a futex.
/// Best effort: a kernel without NUMA (ENOSYS) or a failed call leaves the memory under the
/// default policy, which is what it had without this. Only anonymous memory and private
/// mappings of an ordinary file are bound (see [`numa_policy_is_private`]); guest memory that
/// maps a memfd keeps the policy it has.
pub fn defer_numa_balancing(memory: &GuestMemoryMmap, filter: Arc<crate::seccomp::BpfProgram>) {
    defer_numa_balancing_for(
        memory,
        NUMA_BALANCING_AFTER_SECS.load(std::sync::atomic::Ordering::Relaxed),
        filter,
    )
}

/// Whether `mbind(2)` on `region` changes the policy of this process's mapping alone.
///
/// A mapping of a memfd does not qualify, private or shared. A shmem mapping keeps its policy
/// in the file (`shmem_set_policy`), and every process that maps the file reads it from
/// there: the clones of one UFFD minor-fault handler map one memfd, so a clone binding its
/// mapping would bind all of them, and its timer would return all of them to the default
/// policy. Nothing is lost by leaving such a mapping alone: `task_numa_work` skips a shmem
/// mapping whose file has no policy (`vma_policy_mof`), and it skips every hugetlb mapping.
/// A backing file whose filesystem cannot be read is treated the same way.
fn numa_policy_is_private(region: &GuestRegionMmapExt) -> bool {
    match region.inner.file_offset() {
        None => true,
        Some(file_offset) => BackingFileSystem::of(file_offset.file())
            .is_ok_and(|fs| !fs.is_shmem() && fs.hugetlbfs_page_size().is_none()),
    }
}

fn defer_numa_balancing_for(
    memory: &GuestMemoryMmap,
    secs: u64,
    filter: Arc<crate::seccomp::BpfProgram>,
) {
    if secs == 0 {
        return;
    }
    let ranges: Vec<(usize, usize)> = memory
        .iter()
        .filter(|region| numa_policy_is_private(region))
        .map(|region| (region.as_ptr() as usize, u64_to_usize(region.len())))
        .collect();
    if ranges.is_empty() {
        return;
    }
    for (bound, &(addr, len)) in ranges.iter().enumerate() {
        if let Err(err) = mbind_range(addr, len, libc::MPOL_LOCAL) {
            if err.raw_os_error() != Some(libc::ENOSYS) {
                warn!("guest memory stays in NUMA balancing: mbind(MPOL_LOCAL) failed: {err}");
            }
            for &(addr, len) in &ranges[..bound] {
                let _ = mbind_range(addr, len, libc::MPOL_DEFAULT);
            }
            return;
        }
    }
    let until = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let revert = ranges.clone();
    let spawned = std::thread::Builder::new()
        .name("fc_numa_window".into())
        .spawn(move || {
            if let Err(err) = crate::seccomp::apply_filter(&filter) {
                // Without its filter the thread must not go on; the memory stays out of
                // balancing, as it would be with the exclusion never ending.
                error!("NUMA balancing window: cannot apply the seccomp filter: {err}");
                return;
            }
            // park_timeout waits on a futex, which the VMM's filter allows; it can wake early.
            while let Some(left) = until.checked_duration_since(std::time::Instant::now()) {
                if left.is_zero() {
                    break;
                }
                std::thread::park_timeout(left);
            }
            for (addr, len) in ranges {
                if let Err(err) = mbind_range(addr, len, libc::MPOL_DEFAULT) {
                    warn!("guest memory stays out of NUMA balancing: mbind(MPOL_DEFAULT) failed: {err}");
                }
            }
        });
    if let Err(err) = spawned {
        warn!("guest memory stays in NUMA balancing: cannot start its timer: {err}");
        for (addr, len) in revert {
            let _ = mbind_range(addr, len, libc::MPOL_DEFAULT);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::undocumented_unsafe_blocks)]

    use std::collections::HashMap;
    use std::io::{Read, Seek, Write};
    use std::os::unix::fs::{FileExt, MetadataExt};

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::arch::host_page_size;
    use crate::snapshot::Snapshot;
    use crate::test_utils::single_region_mem;
    use crate::utils::mib_to_bytes;
    use crate::vstate::memory::test_utils::{data_runs, into_region_ext};

    #[test]
    fn test_anonymous() {
        for dirty_page_tracking in [true, false] {
            let region_size = 0x10000;
            let regions = vec![
                (GuestAddress(0x0), region_size),
                (GuestAddress(0x10000), region_size),
                (GuestAddress(0x20000), region_size),
                (GuestAddress(0x30000), region_size),
            ];

            let guest_memory =
                anonymous(&regions, dirty_page_tracking, HugePageConfig::None).unwrap();
            guest_memory.iter().for_each(|region| {
                assert_eq!(region.bitmap().is_some(), dirty_page_tracking);
            });
        }
    }

    #[test]
    fn test_snapshot_file_success() {
        for dirty_page_tracking in [true, false] {
            let page_size = 0x1000;
            let mut file = TempFile::new().unwrap().into_file();
            file.set_len(page_size as u64).unwrap();
            file.write_all(&vec![0x42u8; page_size]).unwrap();

            let regions = vec![(GuestAddress(0), page_size)];
            let guest_regions =
                snapshot_file(file, &regions, dirty_page_tracking, HugePageConfig::None).unwrap();
            assert_eq!(guest_regions.len(), 1);
            guest_regions.iter().for_each(|region| {
                assert_eq!(region.bitmap().is_some(), dirty_page_tracking);
            });
        }
    }

    /// Overlapping ranges whose clamped lengths add up past `u64::MAX` saturate there rather
    /// than wrap to a small total that a bound would pass.
    #[test]
    fn overlapping_write_set_ranges_saturate() {
        let bounds = [(0, u64::MAX)];
        let ranges = [(0, u64::MAX), (0, u64::MAX), (1, u64::MAX)];
        assert_eq!(clamped_len(&bounds, &ranges), u64::MAX);
        assert_eq!(clamped_len(&[(0, 8192)], &[(0, 4096), (4096, 8192)]), 8192);
    }

    /// Populating a write set gives exactly its pages a private copy in this process, before
    /// anything writes them; the rest of the private file mapping stays unfaulted.
    #[test]
    fn populating_a_write_set_copies_exactly_its_pages() {
        let page = 0x1000usize;
        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(&vec![0x5au8; 16 * page]).unwrap();
        let regions = snapshot_file(
            file,
            &[(GuestAddress(0), 16 * page)],
            false,
            HugePageConfig::None,
        )
        .unwrap();
        let ranges = [
            (4 * page as u64, 2 * page as u64),
            (100 * page as u64, page as u64),
        ];
        assert_eq!(
            populate_write_file_ranges(&regions, &ranges).unwrap(),
            2 * page as u64
        );
        let base = regions[0].as_ptr() as u64;
        let mut pagemap = File::open("/proc/self/pagemap").unwrap();
        for index in 0..16u64 {
            let mut entry = [0u8; 8];
            pagemap
                .seek(SeekFrom::Start((base / page as u64 + index) * 8))
                .unwrap();
            pagemap.read_exact(&mut entry).unwrap();
            let entry = u64::from_ne_bytes(entry);
            let private_copy = entry >> 63 & 1 == 1 && entry >> 61 & 1 == 0;
            assert_eq!(
                private_copy,
                (4..6).contains(&index),
                "page {index}: {entry:#x}"
            );
        }
        // The copy holds the file's bytes.
        // SAFETY: the page lies inside the region, which `regions` keeps mapped.
        assert_eq!(unsafe { *regions[0].as_ptr().add(4 * page) }, 0x5a);
    }

    #[test]
    fn test_snapshot_file_multiple_regions() {
        let page_size = 0x1000;
        let total_size = 3 * page_size;
        let mut file = TempFile::new().unwrap().into_file();
        file.set_len(total_size as u64).unwrap();
        file.write_all(&vec![0x42u8; total_size]).unwrap();

        let regions = vec![
            (GuestAddress(0), page_size),
            (GuestAddress(0x10000), page_size),
            (GuestAddress(0x20000), page_size),
        ];
        let guest_regions = snapshot_file(file, &regions, false, HugePageConfig::None).unwrap();
        assert_eq!(guest_regions.len(), 3);
    }

    #[test]
    fn test_snapshot_file_offset_too_large() {
        let page_size = 0x1000;
        let mut file = TempFile::new().unwrap().into_file();
        file.set_len(page_size as u64).unwrap();
        file.write_all(&vec![0x42u8; page_size]).unwrap();

        let regions = vec![(GuestAddress(0), 2 * page_size)];
        let result = snapshot_file(file, &regions, false, HugePageConfig::None);
        assert!(matches!(result.unwrap_err(), MemoryError::OffsetTooLarge));
    }

    #[test]
    fn test_memfd_backing_size_overflow() {
        let mut backing = MemfdBacking::new(0x1000, HugePageConfig::None).unwrap();

        assert!(matches!(
            backing.allocate(&[(GuestAddress(0), 0x1001)], false, HugePageConfig::None),
            Err(MemoryError::OffsetTooLarge)
        ));
    }

    #[test]
    #[should_panic(expected = "regions should be allocated in increasing order")]
    fn test_memfd_backing_guest_address_order_across_calls() {
        let page_size = host_page_size();
        let mut backing = MemfdBacking::new(2 * page_size as u64, HugePageConfig::None).unwrap();

        backing
            .allocate(
                &[(GuestAddress(0x2000), page_size)],
                false,
                HugePageConfig::None,
            )
            .unwrap();
        // Starts below the end of the previous region.
        let _ = backing.allocate(
            &[(GuestAddress(0x1000), page_size)],
            false,
            HugePageConfig::None,
        );
    }

    #[test]
    #[should_panic(expected = "regions should be allocated in increasing order")]
    fn test_memfd_backing_guest_address_order_single_call() {
        let page_size = host_page_size();
        let mut backing = MemfdBacking::new(2 * page_size as u64, HugePageConfig::None).unwrap();

        // Out of order within a single call.
        let _ = backing.allocate(
            &[
                (GuestAddress(0x2000), page_size),
                (GuestAddress(0x1000), page_size),
            ],
            false,
            HugePageConfig::None,
        );
    }

    #[test]
    fn test_memfd_backing_is_a_snapshot_file() {
        let page_size = host_page_size();
        // Two DRAM regions with a gap between them, then a hotplug region.
        let layout = [
            (GuestAddress(0), 2 * page_size),
            (GuestAddress(0x1_0000), 3 * page_size),
            (GuestAddress(0x10_0000), 4 * page_size),
        ];
        let total = 9 * page_size;
        let mut backing = MemfdBacking::new(total as u64, HugePageConfig::None).unwrap();
        let regions = backing
            .allocate(&layout, false, HugePageConfig::None)
            .unwrap();
        let guest_memory = into_region_ext(regions);

        // Each region sits in the memfd where a full snapshot would put it.
        let mut offset = 0;
        for region in guest_memory.iter() {
            let file_offset = region.file_offset().unwrap();
            assert_eq!(file_offset.file().as_raw_fd(), backing.file.as_raw_fd());
            assert_eq!(file_offset.start(), offset);
            offset += region.len();
        }
        assert_eq!(offset, total as u64);

        // So a full snapshot written by Firecracker is byte for byte the memfd.
        for (value, &(addr, size)) in (1u8..).zip(layout.iter()) {
            guest_memory.write(&vec![value; size], addr).unwrap();
        }
        let mut memory_file = TempFile::new().unwrap().into_file();
        guest_memory
            .dump(
                &mut memory_file,
                ZeroPages::Write,
                &guest_memory.take_discarded(),
            )
            .unwrap();
        let mut dumped = Vec::new();
        memory_file.rewind().unwrap();
        memory_file.read_to_end(&mut dumped).unwrap();
        let mut memfd_contents = vec![0u8; total];
        backing.file.read_exact_at(&mut memfd_contents, 0).unwrap();
        assert_eq!(dumped, memfd_contents);
    }

    #[test]
    fn test_mark_dirty_host() {
        let page_size = host_page_size();
        let region_size = page_size * 3;

        let regions = vec![
            (GuestAddress(0), region_size),                  // pages 0-2
            (GuestAddress(region_size as u64), region_size), // pages 3-5
        ];
        let mem = into_region_ext(anonymous(&regions, true, HugePageConfig::None).unwrap());
        let dirty_at = |page: usize| {
            mem.get_slices(GuestAddress((page * page_size) as u64), 1)
                .flatten()
                .all(|slice| slice.bitmap().dirty_at(0))
        };
        let host_ptr = |page: usize, offset: usize| {
            mem.get_slice(GuestAddress((page * page_size) as u64), 1)
                .unwrap()
                .ptr_guard()
                .as_ptr()
                .wrapping_add(offset)
        };

        // Marking through a host pointer lands on the page it maps, and only that one.
        mark_dirty_host(&mem, host_ptr(1, 100), 10);
        assert!(!dirty_at(0));
        assert!(dirty_at(1));
        assert!(!dirty_at(2));

        // The second region has its own bitmap, indexed from its own start.
        mem.reset_dirty();
        mark_dirty_host(&mem, host_ptr(4, 0), page_size + 1);
        assert!(!dirty_at(3));
        assert!(dirty_at(4));
        assert!(dirty_at(5));

        // A range reaching exactly the end of a region is fine.
        mem.reset_dirty();
        mark_dirty_host(&mem, host_ptr(2, page_size - 1), 1);
        assert!(!dirty_at(1));
        assert!(dirty_at(2));
        assert!(!dirty_at(3));

        // Empty ranges mark nothing, wherever they point.
        mem.reset_dirty();
        mark_dirty_host(&mem, host_ptr(1, 0), 0);
        mark_dirty_host(&mem, host_ptr(2, page_size), 0);
        mark_dirty_host(&mem, std::ptr::null(), 0);
        assert!(!(0..6).any(dirty_at));
    }

    #[test]
    #[should_panic(expected = "is not within a single guest memory region")]
    fn test_mark_dirty_host_crossing_region_end() {
        let page_size = host_page_size();
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), page_size)], true, HugePageConfig::None).unwrap(),
        );
        let ptr = mem
            .get_slice(GuestAddress(0), 1)
            .unwrap()
            .ptr_guard()
            .as_ptr();
        mark_dirty_host(&mem, ptr.wrapping_add(page_size - 1), 2);
    }

    #[test]
    #[should_panic(expected = "is not within a single guest memory region")]
    fn test_mark_dirty_host_outside_guest_memory() {
        let page_size = host_page_size();
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), page_size)], true, HugePageConfig::None).unwrap(),
        );
        let host_buf = vec![0u8; page_size];
        mark_dirty_host(&mem, host_buf.as_ptr(), 1);
    }

    #[test]
    fn test_mark_dirty() {
        let page_size = host_page_size();
        let region_size = page_size * 3;

        let regions = vec![
            (GuestAddress(0), region_size),                      // pages 0-2
            (GuestAddress(region_size as u64), region_size),     // pages 3-5
            (GuestAddress(region_size as u64 * 2), region_size), // pages 6-8
        ];
        let guest_memory =
            into_region_ext(anonymous(&regions, true, HugePageConfig::None).unwrap());

        let dirty_map = [
            // page 0: not dirty
            (0, page_size, false),
            // pages 1-2: dirty range in one region
            (page_size, page_size * 2, true),
            // page 3: not dirty
            (page_size * 3, page_size, false),
            // pages 4-7: dirty range across 2 regions,
            (page_size * 4, page_size * 4, true),
            // page 8: not dirty
            (page_size * 8, page_size, false),
        ];

        // Mark dirty memory
        for (addr, len, dirty) in &dirty_map {
            if *dirty {
                guest_memory.mark_dirty(GuestAddress(*addr as u64), *len);
            }
        }

        // Check that the dirty memory was set correctly
        for (addr, len, dirty) in &dirty_map {
            for slice in guest_memory
                .get_slices(GuestAddress(*addr as u64), *len)
                .flatten()
            {
                for i in 0..slice.len() {
                    assert_eq!(slice.bitmap().dirty_at(i), *dirty);
                }
            }
        }
    }

    fn check_serde<M: GuestMemoryExtension>(guest_memory: &M) {
        let original_state = guest_memory.describe();

        // Test direct bitcode serialization
        let serialized_data = bitcode::serialize(&original_state).unwrap();
        let restored_state: GuestMemoryState = bitcode::deserialize(&serialized_data).unwrap();
        assert_eq!(original_state, restored_state);

        // Test with Snapshot wrapper
        let snapshot_data = bitcode::serialize(&Snapshot::new(original_state.clone())).unwrap();
        let restored_snapshot = Snapshot::load_without_crc_check(&snapshot_data).unwrap();
        assert_eq!(original_state, restored_snapshot.data);
    }

    #[test]
    fn test_serde() {
        let page_size = host_page_size();
        let region_size = page_size * 3;

        // Test with a single region
        let guest_memory = into_region_ext(
            anonymous(
                &[(GuestAddress(0), region_size)],
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        check_serde(&guest_memory);

        // Test with some regions
        let regions = vec![
            (GuestAddress(0), region_size),                      // pages 0-2
            (GuestAddress(region_size as u64), region_size),     // pages 3-5
            (GuestAddress(region_size as u64 * 2), region_size), // pages 6-8
        ];
        let guest_memory =
            into_region_ext(anonymous(&regions, true, HugePageConfig::None).unwrap());
        check_serde(&guest_memory);
    }

    #[test]
    fn test_describe() {
        let page_size: usize = host_page_size();

        // Two regions of one page each, with a one page gap between them.
        let mem_regions = [
            (GuestAddress(0), page_size),
            (GuestAddress(page_size as u64 * 2), page_size),
        ];
        let guest_memory =
            into_region_ext(anonymous(&mem_regions, true, HugePageConfig::None).unwrap());

        let expected_memory_state = GuestMemoryState {
            regions: vec![
                GuestMemoryRegionState {
                    base_address: 0,
                    size: page_size,
                    region_type: GuestRegionType::Dram,
                    plugged: vec![true],
                },
                GuestMemoryRegionState {
                    base_address: page_size as u64 * 2,
                    size: page_size,
                    region_type: GuestRegionType::Dram,
                    plugged: vec![true],
                },
            ],
        };

        let actual_memory_state = guest_memory.describe();
        assert_eq!(expected_memory_state, actual_memory_state);

        // Two regions of three pages each, with a one page gap between them.
        let mem_regions = [
            (GuestAddress(0), page_size * 3),
            (GuestAddress(page_size as u64 * 4), page_size * 3),
        ];
        let guest_memory =
            into_region_ext(anonymous(&mem_regions, true, HugePageConfig::None).unwrap());

        let expected_memory_state = GuestMemoryState {
            regions: vec![
                GuestMemoryRegionState {
                    base_address: 0,
                    size: page_size * 3,
                    region_type: GuestRegionType::Dram,
                    plugged: vec![true],
                },
                GuestMemoryRegionState {
                    base_address: page_size as u64 * 4,
                    size: page_size * 3,
                    region_type: GuestRegionType::Dram,
                    plugged: vec![true],
                },
            ],
        };

        let actual_memory_state = guest_memory.describe();
        assert_eq!(expected_memory_state, actual_memory_state);
    }

    #[test]
    fn test_dump() {
        let page_size = host_page_size();

        // Two regions of two pages each, with a one page gap between them.
        let region_1_address = GuestAddress(0);
        let region_2_address = GuestAddress(page_size as u64 * 3);
        let region_size = page_size * 2;
        let mem_regions = [
            (region_1_address, region_size),
            (region_2_address, region_size),
        ];
        let guest_memory =
            into_region_ext(anonymous(&mem_regions, true, HugePageConfig::None).unwrap());
        // Check that Firecracker bitmap is clean.
        guest_memory.iter().for_each(|r| {
            assert!(!r.bitmap().dirty_at(0));
            assert!(!r.bitmap().dirty_at(1));
        });

        // Fill the first region with 1s and the second with 2s.
        let first_region = vec![1u8; region_size];
        guest_memory.write(&first_region, region_1_address).unwrap();

        let second_region = vec![2u8; region_size];
        guest_memory
            .write(&second_region, region_2_address)
            .unwrap();

        let memory_state = guest_memory.describe();

        // dump the full memory.
        let mut memory_file = TempFile::new().unwrap().into_file();
        guest_memory
            .dump(
                &mut memory_file,
                ZeroPages::Write,
                &guest_memory.take_discarded(),
            )
            .unwrap();

        let restored_guest_memory = into_region_ext(
            snapshot_file(
                memory_file,
                &memory_state.regions(),
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        );

        // Check that the region contents are the same.
        let mut restored_region = vec![0u8; page_size * 2];
        restored_guest_memory
            .read(restored_region.as_mut_slice(), region_1_address)
            .unwrap();
        assert_eq!(first_region, restored_region);

        restored_guest_memory
            .read(restored_region.as_mut_slice(), region_2_address)
            .unwrap();
        assert_eq!(second_region, restored_region);
    }

    #[test]
    fn test_hole_granule_is_fcvm_working_set_granule() {
        // fcvm's working_set::GRANULE (src/uffd/working_set.rs in fcvm) is 4096, and fcvm reads
        // each hole in a memory file as whole granules of that size. Change both or neither.
        assert_eq!(HOLE_GRANULE, 4096);
    }

    #[test]
    fn test_dump_leaves_zero_pages_as_holes() {
        let page = HOLE_GRANULE;
        // Region 1 holds [ones, zero, zero, a page whose last byte is two, zero]. It ends in a
        // zero page, so region 2 lands at the right file offset only if that page is seeked
        // over. Region 2 holds [zero, threes, threes, zero] and ends the file in a zero page.
        // The zero pages are never written, so AbsentPages::Skip leaves them out unread where
        // the host page is no larger than a granule.
        let regions = [
            (GuestAddress(0), page * 5),
            (GuestAddress(page as u64 * 8), page * 4),
        ];
        for absent_pages in [AbsentPages::Read, AbsentPages::Skip] {
            let guest_memory =
                into_region_ext(anonymous(&regions, false, HugePageConfig::None).unwrap());
            guest_memory
                .write_slice(&vec![1u8; page], GuestAddress(0))
                .unwrap();
            guest_memory
                .write_slice(&[2u8], GuestAddress(page as u64 * 4 - 1))
                .unwrap();
            guest_memory
                .write_slice(&vec![3u8; page * 2], GuestAddress(page as u64 * 9))
                .unwrap();
            let memory_state = guest_memory.describe();

            // The file starts empty and at its full length, as snapshot_memory_to_path leaves it.
            let total = page as u64 * 9;
            let mut memory_file = TempFile::new().unwrap().into_file();
            memory_file.set_len(total).unwrap();
            guest_memory
                .dump(
                    &mut memory_file,
                    ZeroPages::Hole(absent_pages),
                    &guest_memory.take_discarded(),
                )
                .unwrap();
            assert_eq!(
                memory_file.stream_position().unwrap(),
                total,
                "{absent_pages:?}"
            );

            // Only the pages holding a non-zero byte are data.
            let page = page as u64;
            assert_eq!(
                data_runs(&memory_file),
                vec![(0, page), (page * 3, page), (page * 6, page * 2)],
                "{absent_pages:?}"
            );

            // Every byte reads back as it was in guest memory.
            let restored_guest_memory = into_region_ext(
                snapshot_file(
                    memory_file,
                    &memory_state.regions(),
                    false,
                    HugePageConfig::None,
                )
                .unwrap(),
            );
            for (address, size) in regions {
                let mut expected = vec![0u8; size];
                guest_memory.read_slice(&mut expected, address).unwrap();
                let mut restored = vec![0xAAu8; size];
                restored_guest_memory
                    .read_slice(&mut restored, address)
                    .unwrap();
                assert_eq!(
                    expected, restored,
                    "{absent_pages:?}: region at {address:?}"
                );
            }
        }
    }

    // Dumps guest memory mapped from a file that was then cut to half its length, as when the
    // file a guest was restored from is truncated under it. Reading the second half raises
    // SIGBUS. Returns the error the dump reports.
    fn dump_memory_of_truncated_file(zero_pages: ZeroPages) -> MemoryError {
        let size = host_page_size() * 8;
        let mut backing = TempFile::new().unwrap().into_file();
        backing.write_all(&vec![0x11u8; size]).unwrap();
        let truncate = backing.try_clone().unwrap();
        let guest_memory = into_region_ext(
            snapshot_file(
                backing,
                &[(GuestAddress(0), size)],
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        truncate.set_len(size as u64 / 2).unwrap();

        let mut memory_file = TempFile::new().unwrap().into_file();
        memory_file.set_len(size as u64).unwrap();
        guest_memory
            .dump(&mut memory_file, zero_pages, &guest_memory.take_discarded())
            .unwrap_err()
    }

    #[test]
    fn test_dump_write_reports_unreadable_guest_memory() {
        let err = dump_memory_of_truncated_file(ZeroPages::Write);
        assert!(err.to_string().starts_with("Cannot dump memory: "), "{err}");
    }

    #[test]
    fn test_dump_hole_reports_unreadable_guest_memory() {
        let err = dump_memory_of_truncated_file(ZeroPages::Hole(AbsentPages::Read));
        assert!(err.to_string().starts_with("Cannot dump memory: "), "{err}");
    }

    // A writer whose every write fails, as on a full disk. Seeks go to a file.
    struct FailingWriter(File);

    impl WriteVolatile for FailingWriter {
        fn write_volatile<B: BitmapSlice>(
            &mut self,
            _buf: &VolatileSlice<B>,
        ) -> Result<usize, vm_memory::VolatileMemoryError> {
            Err(vm_memory::VolatileMemoryError::IOError(
                io::Error::from_raw_os_error(libc::ENOSPC),
            ))
        }
    }

    impl Seek for FailingWriter {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.0.seek(pos)
        }
    }

    #[test]
    fn test_dump_reports_a_failed_write_in_both_modes() {
        let page = HOLE_GRANULE;
        // Guest memory of two pages, one of them all zero. With the zero page second, a Hole dump
        // writes the first page when its scan reaches the zero page; with the zero page first, it
        // writes the second page at the end of the chunk. A Write dump writes the whole slot.
        for zero_page in [1, 0] {
            for zero_pages in [
                ZeroPages::Write,
                ZeroPages::Hole(AbsentPages::Read),
                ZeroPages::Hole(AbsentPages::Skip),
            ] {
                let guest_memory = into_region_ext(
                    anonymous(&[(GuestAddress(0), page * 2)], false, HugePageConfig::None).unwrap(),
                );
                let data_page = 1 - zero_page;
                guest_memory
                    .write_slice(&vec![0x5Au8; page], GuestAddress((data_page * page) as u64))
                    .unwrap();

                let mut writer = FailingWriter(TempFile::new().unwrap().into_file());
                let err = guest_memory
                    .dump(&mut writer, zero_pages, &guest_memory.take_discarded())
                    .unwrap_err();
                assert!(
                    err.to_string().starts_with("Cannot dump memory: "),
                    "{zero_pages:?}, zero page {zero_page}: {err}"
                );
            }
        }
    }

    // A writer whose every seek fails. Writes go to a file.
    struct FailingSeeker(File);

    impl WriteVolatile for FailingSeeker {
        fn write_volatile<B: BitmapSlice>(
            &mut self,
            buf: &VolatileSlice<B>,
        ) -> Result<usize, vm_memory::VolatileMemoryError> {
            self.0.write_volatile(buf)
        }
    }

    impl Seek for FailingSeeker {
        fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
            Err(io::Error::from_raw_os_error(libc::ESPIPE))
        }
    }

    /// Tracked guest memory of `len` bytes that is one slot of a hotpluggable region.
    fn hotplug_slot_memory(len: usize, plugged: bool) -> GuestMemoryMmap {
        let region = anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::None).unwrap();
        let state = GuestMemoryRegionState {
            base_address: 0,
            size: len,
            region_type: GuestRegionType::Hotpluggable,
            plugged: vec![plugged],
        };
        let region =
            GuestRegionMmapExt::from_state(region.into_iter().next().unwrap(), &state, 0).unwrap();
        GuestMemoryMmap::from_regions(vec![region]).unwrap()
    }

    #[test]
    fn test_dumps_return_a_failed_seek() {
        let len = 2 * host_page_size();

        // A Full dump seeks over an unplugged slot.
        for zero_pages in [ZeroPages::Write, ZeroPages::Hole(AbsentPages::Skip)] {
            let mut writer = FailingSeeker(TempFile::new().unwrap().into_file());
            let memory = hotplug_slot_memory(len, false);
            let err = memory
                .dump(&mut writer, zero_pages, &memory.take_discarded())
                .unwrap_err();
            assert!(
                matches!(err, MemoryError::SeekError(_)),
                "{zero_pages:?}: {err}"
            );
        }

        // A Diff dump seeks over the clean second page, the last of its slot.
        let mut writer = FailingSeeker(TempFile::new().unwrap().into_file());
        let err = hotplug_slot_memory(len, true)
            .dump_dirty(&mut writer, &vec![vec![0b01]], &vec![vec![0]])
            .unwrap_err();
        assert!(matches!(err, MemoryError::SeekError(_)), "{err}");
    }

    #[test]
    fn test_is_zero_reads_unaligned_head_and_tail_bytes() {
        // u64 storage starts the buffer on an 8-byte boundary, so a slice from byte 3 has a
        // 5-byte head before its first aligned word. 3 + 4091 is not a multiple of 8, so the
        // slice also ends in a tail after its last aligned word.
        let (start, len) = (3, 4091);
        let mut storage = vec![0u64; 4096 / 8];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast::<u8>(), storage.len() * 8)
        };
        let head = bytes[start..].as_ptr().align_offset(8);
        assert!(head > 0 && (len - head) % 8 > 0, "head {head}");

        assert!(is_zero(&VolatileSlice::from(
            &mut bytes[start..start + len]
        )));

        bytes[start] = 1;
        assert!(
            !is_zero(&VolatileSlice::from(&mut bytes[start..start + len])),
            "non-zero first byte, in the head"
        );
        bytes[start] = 0;

        bytes[start + len - 1] = 1;
        assert!(
            !is_zero(&VolatileSlice::from(&mut bytes[start..start + len])),
            "non-zero last byte, in the tail"
        );
    }

    // A file writer that records the length of every write a dump asks for.
    struct RecordingWriter {
        file: File,
        writes: Vec<usize>,
    }

    impl WriteVolatile for RecordingWriter {
        fn write_volatile<B: BitmapSlice>(
            &mut self,
            buf: &VolatileSlice<B>,
        ) -> Result<usize, vm_memory::VolatileMemoryError> {
            self.writes.push(buf.len());
            self.file.write_volatile(buf)
        }
    }

    impl Seek for RecordingWriter {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.file.seek(pos)
        }
    }

    #[test]
    fn test_dump_hole_writes_at_most_one_chunk_at_a_time() {
        let (page, chunk) = (HOLE_GRANULE, DUMP_CHUNK);
        // [zero page, a non-zero run longer than a chunk that crosses the first chunk boundary,
        // a zero run that crosses the second, one non-zero page, one trailing zero page]. Only
        // the non-zero runs are written to guest memory, so with AbsentPages::Skip the zero runs
        // are scanned where a transparent huge page backs them and left out unread elsewhere.
        let size = chunk * 2 + page * 3;
        let mut contents = vec![0u8; size];
        contents[page..chunk + page * 2].fill(0x5A);
        contents[chunk * 2 + page..chunk * 2 + page * 2].fill(0x5A);
        for absent_pages in [AbsentPages::Read, AbsentPages::Skip] {
            let guest_memory = into_region_ext(
                anonymous(&[(GuestAddress(0), size)], false, HugePageConfig::None).unwrap(),
            );
            guest_memory
                .write_slice(&contents[page..chunk + page * 2], GuestAddress(page as u64))
                .unwrap();
            guest_memory
                .write_slice(
                    &contents[chunk * 2 + page..chunk * 2 + page * 2],
                    GuestAddress((chunk * 2 + page) as u64),
                )
                .unwrap();

            let mut writer = RecordingWriter {
                file: TempFile::new().unwrap().into_file(),
                writes: Vec::new(),
            };
            writer.file.set_len(size as u64).unwrap();
            guest_memory
                .dump(
                    &mut writer,
                    ZeroPages::Hole(absent_pages),
                    &guest_memory.take_discarded(),
                )
                .unwrap();

            // A long run is written in pieces of at most a chunk, each soon after it is scanned.
            assert!(
                writer.writes.iter().all(|&len| len <= chunk),
                "{absent_pages:?}: writes {:?}",
                writer.writes
            );
            assert_eq!(
                writer.file.stream_position().unwrap(),
                size as u64,
                "{absent_pages:?}"
            );
            let (page, chunk) = (page as u64, chunk as u64);
            assert_eq!(
                data_runs(&writer.file),
                vec![(page, chunk + page), (chunk * 2 + page, page)],
                "{absent_pages:?}"
            );
            let mut written = Vec::new();
            writer.file.rewind().unwrap();
            writer.file.read_to_end(&mut written).unwrap();
            assert!(
                written == contents,
                "{absent_pages:?}: the file differs from guest memory"
            );
        }
    }

    /// Whether each host page of the `len` bytes of guest memory at `addr` is resident, by
    /// mincore(2). Asking does not fault the pages in.
    fn resident(mem: &GuestMemoryMmap, addr: GuestAddress, len: usize) -> Vec<bool> {
        let page_size = host_page_size();
        let host = mem.get_host_address(addr).unwrap();
        let mut pages = vec![0u8; len / page_size];
        let ret = unsafe { libc::mincore(host.cast(), len, pages.as_mut_ptr()) };
        assert_eq!(ret, 0, "mincore: {}", io::Error::last_os_error());
        pages.iter().map(|page| page & 1 != 0).collect()
    }

    #[test]
    fn test_dump_skipping_absent_pages_leaves_them_unmapped() {
        // AbsentPages::Skip must neither page in nor read a page of anonymous memory that has no
        // page table entry: either would map the zero page there. Four host pages cannot hold a
        // transparent huge page, so the write to the second maps that page alone.
        let page_size = host_page_size();
        let len = 4 * page_size;
        let guest_memory = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], false, HugePageConfig::None).unwrap(),
        );
        guest_memory
            .write_slice(&vec![0x77u8; page_size], GuestAddress(page_size as u64))
            .unwrap();
        assert_eq!(
            resident(&guest_memory, GuestAddress(0), len),
            [false, true, false, false],
            "test setup"
        );

        let mut memory_file = TempFile::new().unwrap().into_file();
        memory_file.set_len(len as u64).unwrap();
        guest_memory
            .dump(
                &mut memory_file,
                ZeroPages::Hole(AbsentPages::Skip),
                &guest_memory.take_discarded(),
            )
            .unwrap();

        assert_eq!(
            resident(&guest_memory, GuestAddress(0), len),
            [false, true, false, false],
            "the dump mapped pages that had no page table entry"
        );
        let mut expected = vec![0u8; len];
        expected[page_size..2 * page_size].fill(0x77);
        let mut dumped = Vec::new();
        memory_file.rewind().unwrap();
        memory_file.read_to_end(&mut dumped).unwrap();
        assert!(dumped == expected, "the file differs from guest memory");

        // AbsentPages::Read pages every page in to scan it.
        memory_file.rewind().unwrap();
        guest_memory
            .dump(
                &mut memory_file,
                ZeroPages::Hole(AbsentPages::Read),
                &guest_memory.take_discarded(),
            )
            .unwrap();
        assert_eq!(
            resident(&guest_memory, GuestAddress(0), len),
            [true; 4],
            "AbsentPages::Read"
        );
    }

    #[test]
    fn test_dump_skipping_absent_pages_reads_file_backed_regions() {
        // A page of a file-backed region that the VMM never touched reads as the file's data, so
        // AbsentPages::Skip must not leave it as a hole. The file's pages are dropped from the
        // page cache first, so neither pagemap nor mincore reports them, and only the region's
        // file backing keeps the dump from skipping them.
        let page_size = host_page_size();
        let region_size = page_size * 4;
        let mut source = TempFile::new().unwrap().into_file();
        source.write_all(&vec![0x77u8; region_size]).unwrap();
        source.sync_all().unwrap();
        let ret =
            unsafe { libc::posix_fadvise(source.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        assert_eq!(ret, 0);
        let guest_memory = into_region_ext(
            snapshot_file(
                source,
                &[(GuestAddress(0), region_size)],
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        assert_eq!(
            resident(&guest_memory, GuestAddress(0), region_size),
            [false; 4],
            "test setup: the file's pages must not be in the page cache; is the temporary \
             directory on tmpfs?"
        );

        let mut memory_file = TempFile::new().unwrap().into_file();
        guest_memory
            .dump(
                &mut memory_file,
                ZeroPages::Hole(AbsentPages::Skip),
                &guest_memory.take_discarded(),
            )
            .unwrap();

        let mut dumped = Vec::new();
        memory_file.rewind().unwrap();
        memory_file.read_to_end(&mut dumped).unwrap();
        assert!(
            dumped == vec![0x77u8; region_size],
            "the dump left {} of {region_size} bytes of the file-backed region as holes",
            region_size - dumped.len().min(region_size)
        );
    }

    #[test]
    fn test_dump_dirty() {
        let page_size = host_page_size();

        // Two regions of two pages each, with a one page gap between them.
        let region_1_address = GuestAddress(0);
        let region_2_address = GuestAddress(page_size as u64 * 3);
        let region_size = page_size * 2;
        let mem_regions = [
            (region_1_address, region_size),
            (region_2_address, region_size),
        ];
        let guest_memory =
            into_region_ext(anonymous(&mem_regions, true, HugePageConfig::None).unwrap());
        // Check that Firecracker bitmap is clean.
        guest_memory.iter().for_each(|r| {
            assert!(!r.bitmap().dirty_at(0));
            assert!(!r.bitmap().dirty_at(1));
        });

        // Fill the first region with 1s and the second with 2s.
        let first_region = vec![1u8; region_size];
        guest_memory.write(&first_region, region_1_address).unwrap();

        let second_region = vec![2u8; region_size];
        guest_memory
            .write(&second_region, region_2_address)
            .unwrap();

        // Firecracker Dirty Bitmap after the writes:
        // First region pages: [dirty, dirty]
        // Second region pages: [dirty, dirty]

        let memory_state = guest_memory.describe();

        // KVM dirty bitmap:
        // First region pages: [dirty, clean]
        // Second region pages: [clean, dirty]
        let mut kvm_logs = vec![vec![0b01], vec![0b10]];

        let take = |kvm_logs: &[Vec<u64>]| {
            let mut dirty_bitmap = guest_memory.take_dirty();
            guest_memory
                .merge_slot_kvm_dirty_logs(&mut dirty_bitmap, |mem_slot| {
                    Ok::<_, ()>(kvm_logs[mem_slot.slot as usize].clone())
                })
                .unwrap();
            dirty_bitmap
        };

        let mut file = TempFile::new().unwrap().into_file();
        guest_memory
            .dump_dirty(&mut file, &take(&kvm_logs), &guest_memory.take_discarded())
            .unwrap();

        // We can restore from this because this is the first dirty dump.
        let restored_guest_memory = into_region_ext(
            snapshot_file(file, &memory_state.regions(), false, HugePageConfig::None).unwrap(),
        );

        // Check that the region contents are the same.
        let mut restored_region = vec![0u8; region_size];
        restored_guest_memory
            .read(restored_region.as_mut_slice(), region_1_address)
            .unwrap();
        assert_eq!(first_region, restored_region);

        restored_guest_memory
            .read(restored_region.as_mut_slice(), region_2_address)
            .unwrap();
        assert_eq!(second_region, restored_region);

        // Dirty the memory and dump again
        let file = TempFile::new().unwrap();
        let mut reader = file.into_file();
        let zeros = vec![0u8; page_size];
        let ones = vec![1u8; page_size];
        let twos = vec![2u8; page_size];

        // Firecracker Dirty Bitmap:
        // First region pages: [clean, dirty]
        // Second region pages: [clean, clean]
        guest_memory
            .write(&twos, GuestAddress(page_size as u64))
            .unwrap();
        // KVM dirty bitmap:
        // First region pages: [dirty, clean]
        // Second region pages: [clean, dirty]
        kvm_logs = vec![vec![0b01], vec![0b10]];

        guest_memory
            .dump_dirty(
                &mut reader,
                &take(&kvm_logs),
                &guest_memory.take_discarded(),
            )
            .unwrap();

        // Check that only the dirty regions are dumped.
        let mut diff_file_content = Vec::new();
        let expected_file_contents = [
            ones.as_slice(),
            twos.as_slice(),
            zeros.as_slice(),
            twos.as_slice(),
        ]
        .concat();
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_to_end(&mut diff_file_content).unwrap();
        assert_eq!(expected_file_contents, diff_file_content);

        // Take a 3rd snapshot

        // Firecracker Dirty Bitmap:
        // First region pages: [dirty, clean]
        // Second region pages: [dirty, clean]
        guest_memory.write(&twos, region_1_address).unwrap();
        guest_memory.write(&ones, region_2_address).unwrap();
        // KVM dirty bitmap:
        // First region pages: [clean, clean]
        // Second region pages: [clean, clean]
        kvm_logs = vec![vec![0b00], vec![0b00]];

        let file = TempFile::new().unwrap();
        let logical_size = page_size as u64 * 4;
        file.as_file().set_len(logical_size).unwrap();

        let mut reader = file.into_file();
        guest_memory
            .dump_dirty(
                &mut reader,
                &take(&kvm_logs),
                &guest_memory.take_discarded(),
            )
            .unwrap();

        // Check that only the dirty regions are dumped.
        let mut diff_file_content = Vec::new();
        // The resulting file is a sparse file with holes.
        let expected_file_contents = [
            twos.as_slice(),
            zeros.as_slice(), // hole
            ones.as_slice(),
            zeros.as_slice(), // hole
        ]
        .concat();
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_to_end(&mut diff_file_content).unwrap();

        assert_eq!(expected_file_contents, diff_file_content);

        // Make sure that only 2 of the pages are written in the file and the
        // other two are holes.
        let metadata = reader.metadata().unwrap();
        let physical_size = metadata.blocks() * 512;
        assert_eq!(physical_size, 2 * page_size as u64);
        assert_ne!(physical_size, logical_size);
    }

    #[test]
    fn test_dump_dirty_with_concurrent_writes() {
        let page_size = host_page_size();
        let pages = 64;
        let guest_memory = into_region_ext(
            anonymous(
                &[(GuestAddress(0), pages * page_size)],
                true,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        let mut file = TempFile::new().unwrap().into_file();
        guest_memory
            .dump(&mut file, ZeroPages::Write, &guest_memory.take_discarded())
            .unwrap();
        let diff = |file: &mut File| {
            let dirty_bitmap = guest_memory.take_dirty();
            file.rewind().unwrap();
            guest_memory
                .dump_dirty(file, &dirty_bitmap, &guest_memory.take_discarded())
                .unwrap();
        };

        // A device keeps writing random pages while Diff snapshots are taken.
        let started = std::sync::Barrier::new(2);
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                started.wait();
                let mut version = 0u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    version += 1;
                    let page = vmm_sys_util::rand::xor_pseudo_rng_u32() as usize % pages;
                    guest_memory
                        .write_obj(version, GuestAddress((page * page_size) as u64))
                        .unwrap();
                }
            });
            started.wait();
            for _ in 0..100 {
                diff(&mut file);
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });

        // With the device stopped, one more snapshot must bring the file up to date.
        diff(&mut file);
        let mut memory = vec![0u8; pages * page_size];
        guest_memory.read(&mut memory, GuestAddress(0)).unwrap();
        let mut snapshot = vec![0u8; pages * page_size];
        file.read_exact_at(&mut snapshot, 0).unwrap();
        let stale: Vec<usize> = (0..pages)
            .filter(|page| {
                let range = page * page_size..(page + 1) * page_size;
                memory[range.clone()] != snapshot[range]
            })
            .collect();
        assert!(stale.is_empty(), "stale pages in the snapshot: {stale:?}");
    }

    #[test]
    fn test_take_dirty_with_concurrent_marks() {
        let page_size = host_page_size();
        let pages = 1 << 16;
        let guest_memory = into_region_ext(
            anonymous(
                &[(GuestAddress(0), pages * page_size)],
                true,
                HugePageConfig::None,
            )
            .unwrap(),
        );
        let take = || guest_memory.take_dirty().remove(0);

        // Each page is marked once, so a mark lost by a take never shows up in a later one.
        let run = || {
            let started = std::sync::Barrier::new(2);
            let marking = std::sync::atomic::AtomicBool::new(true);
            let mut taken = vec![0u64; pages / 64];
            let mut takes = 0;
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    started.wait();
                    for page in 0..pages {
                        guest_memory.mark_dirty(GuestAddress((page * page_size) as u64), 1);
                    }
                    marking.store(false, std::sync::atomic::Ordering::Release);
                });
                started.wait();
                while marking.load(std::sync::atomic::Ordering::Acquire) {
                    takes += 1;
                    for (all, word) in taken.iter_mut().zip(take()) {
                        *all |= word;
                    }
                }
            });
            for (all, word) in taken.iter_mut().zip(take()) {
                *all |= word;
            }
            let lost: Vec<usize> = (0..pages)
                .filter(|page| taken[page / 64] & (1 << (page % 64)) == 0)
                .collect();
            assert!(
                lost.is_empty(),
                "{} marks lost, first {:?}",
                lost.len(),
                lost.first()
            );
            takes
        };
        // A run whose takes missed the marking proves nothing, so it is repeated.
        assert!(
            (0..100).any(|_| run() > 1),
            "the takes never overlapped the marking"
        );
    }

    #[test]
    fn test_store_dirty_bitmap() {
        let page_size = host_page_size();

        // Two regions of three pages each, with a one page gap between them.
        let region_1_address = GuestAddress(0);
        let region_2_address = GuestAddress(page_size as u64 * 4);
        let region_size = page_size * 3;
        let mem_regions = [
            (region_1_address, region_size),
            (region_2_address, region_size),
        ];
        let guest_memory =
            into_region_ext(anonymous(&mem_regions, true, HugePageConfig::None).unwrap());

        // Check that Firecracker bitmap is clean.
        guest_memory.iter().for_each(|r| {
            assert!(!r.bitmap().dirty_at(0));
            assert!(!r.bitmap().dirty_at(page_size));
            assert!(!r.bitmap().dirty_at(page_size * 2));
        });

        guest_memory.store_dirty_bitmap(&vec![vec![0b101], vec![0b101]]);

        // Assert that the bitmap now reports as being dirty maching the dirty bitmap
        guest_memory.iter().for_each(|r| {
            assert!(r.bitmap().dirty_at(0));
            assert!(!r.bitmap().dirty_at(page_size));
            assert!(r.bitmap().dirty_at(page_size * 2));
        });
    }

    #[test]
    fn test_merge_slot_kvm_dirty_logs_keeps_logs_read_before_a_failure() {
        let page_size = host_page_size();
        let size = 3 * 64 * page_size;
        let region = anonymous(&[(GuestAddress(0), size)], true, HugePageConfig::None).unwrap();
        let state = GuestMemoryRegionState {
            base_address: 0,
            size,
            region_type: GuestRegionType::Hotpluggable,
            plugged: vec![true, true, true],
        };
        let region =
            GuestRegionMmapExt::from_state(region.into_iter().next().unwrap(), &state, 0).unwrap();
        let guest_memory = GuestMemoryMmap::from_regions(vec![region]).unwrap();

        let mut dirty_bitmap = guest_memory.take_dirty();
        let mut read = Vec::new();
        let result = guest_memory.merge_slot_kvm_dirty_logs(&mut dirty_bitmap, |mem_slot| {
            read.push(mem_slot.slot);
            match mem_slot.slot {
                0 => Ok(vec![0b1]),
                slot => Err(slot),
            }
        });

        assert_eq!(result, Err(1));
        assert_eq!(read, [0, 1]);
        assert_eq!(dirty_bitmap, [vec![0b1, 0, 0]]);
    }

    #[test]
    fn test_create_memfd() {
        let size_bytes = mib_to_bytes(1) as u64;

        let memfd = create_memfd(size_bytes, None).unwrap();

        assert_eq!(memfd.as_file().metadata().unwrap().len(), size_bytes);
        memfd.as_file().set_len(0x69).unwrap_err();

        let mut seals = memfd::SealsHashSet::new();
        seals.insert(memfd::FileSeal::SealGrow);
        memfd.add_seals(&seals).unwrap_err();
    }

    /// This asserts that $lhs matches $rhs.
    macro_rules! assert_match {
        ($lhs:expr, $rhs:pat) => {{ assert!(matches!($lhs, $rhs)) }};
    }

    #[test]
    fn test_discard_range() {
        let page_size: usize = 0x1000;
        let mem = single_region_mem(2 * page_size);

        // Fill the memory with ones.
        let ones = vec![1u8; 2 * page_size];
        mem.write(&ones[..], GuestAddress(0)).unwrap();

        // Remove the first page.
        mem.discard_range(GuestAddress(0), page_size).unwrap();

        // Check that the first page is zeroed.
        let mut actual_page = vec![0u8; page_size];
        mem.read(actual_page.as_mut_slice(), GuestAddress(0))
            .unwrap();
        assert_eq!(vec![0u8; page_size], actual_page);
        // Check that the second page still contains ones.
        mem.read(actual_page.as_mut_slice(), GuestAddress(page_size as u64))
            .unwrap();
        assert_eq!(vec![1u8; page_size], actual_page);

        // Malformed range: the len is too big.
        assert_match!(
            mem.discard_range(GuestAddress(0), 0x10000).unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        // Region not mapped.
        assert_match!(
            mem.discard_range(GuestAddress(0x10000), 0x10).unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        // Unaligned address is rejected.
        assert_match!(
            mem.discard_range(GuestAddress(0x20), page_size)
                .unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        // Range running past the end of the region.
        let region = mem.iter().next().unwrap();
        assert_match!(
            region
                .discard_range(
                    MemoryRegionAddress(page_size as u64),
                    2 * page_size,
                    PartialPages::Zero,
                )
                .unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );
    }

    #[test]
    fn test_discard_range_on_file() {
        let page_size: usize = 0x1000;
        let mut memory_file = TempFile::new().unwrap().into_file();
        memory_file.set_len(2 * page_size as u64).unwrap();
        memory_file.write_all(&vec![2u8; 2 * page_size]).unwrap();
        let mem = into_region_ext(
            snapshot_file(
                memory_file,
                &[(GuestAddress(0), 2 * page_size)],
                false,
                HugePageConfig::None,
            )
            .unwrap(),
        );

        // Fill the memory with ones.
        let ones = vec![1u8; 2 * page_size];
        mem.write(&ones[..], GuestAddress(0)).unwrap();

        // Remove the first page.
        mem.discard_range(GuestAddress(0), page_size).unwrap();

        // Check that the first page is zeroed.
        let mut actual_page = vec![0u8; page_size];
        mem.read(actual_page.as_mut_slice(), GuestAddress(0))
            .unwrap();
        assert_eq!(vec![0u8; page_size], actual_page);
        // Check that the second page still contains ones.
        mem.read(actual_page.as_mut_slice(), GuestAddress(page_size as u64))
            .unwrap();
        assert_eq!(vec![1u8; page_size], actual_page);

        // Malformed range: the len is too big.
        assert_match!(
            mem.discard_range(GuestAddress(0), 0x10000).unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        // Region not mapped.
        assert_match!(
            mem.discard_range(GuestAddress(0x10000), 0x10).unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        // Unaligned address is rejected, so the file-backed branch never sees a failing mmap.
        assert_match!(
            mem.discard_range(GuestAddress(0x20), page_size)
                .unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );
    }

    #[test]
    fn test_discard_range_on_hugetlb_zeroes_partial_pages() {
        if free_hugepages_2m() < 2 {
            println!("Skipping: fewer than two free 2 MiB hugepages available");
            return;
        }
        let page_size = HugePageConfig::Hugetlbfs2M.page_size();
        let mem = into_region_ext(
            anonymous(
                &[(GuestAddress(0), 2 * page_size)],
                false,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        mem.write(&vec![1; 2 * page_size], GuestAddress(0)).unwrap();

        mem.discard_range(GuestAddress(0), host_page_size())
            .unwrap();
        mem.discard_range(GuestAddress(host_page_size() as u64), page_size)
            .unwrap();
        assert_match!(
            mem.discard_range(GuestAddress(0x20), host_page_size())
                .unwrap_err(),
            GuestMemoryError::InvalidGuestAddress(_)
        );

        let discarded_len = page_size + host_page_size();
        let mut actual = vec![0; 2 * page_size];
        mem.read(&mut actual, GuestAddress(0)).unwrap();
        let expected = [
            vec![0; discarded_len],
            vec![1; 2 * page_size - discarded_len],
        ]
        .concat();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_discard_range_on_hugetlb_skips_unplugged_slots() {
        if free_hugepages_2m() < 2 {
            println!("Skipping: fewer than two free 2 MiB hugepages available");
            return;
        }
        let page_size = HugePageConfig::Hugetlbfs2M.page_size();
        let mmap_region = anonymous(
            &[(GuestAddress(0), 2 * page_size)],
            false,
            HugePageConfig::Hugetlbfs2M,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let region = GuestRegionMmapExt::hotpluggable_from_mmap_region(mmap_region, 0, page_size);
        region.plugged.lock().unwrap().set(0, true);
        region.mem_slot(1).protect(true).unwrap();
        region
            .write_slice(&vec![1; page_size], MemoryRegionAddress(0))
            .unwrap();

        region
            .discard_range(
                MemoryRegionAddress(host_page_size() as u64),
                page_size,
                PartialPages::Zero,
            )
            .unwrap();

        let mut actual = vec![0; page_size];
        region
            .read_slice(&mut actual, MemoryRegionAddress(0))
            .unwrap();
        let expected = [
            vec![1; host_page_size()],
            vec![0; page_size - host_page_size()],
        ]
        .concat();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_discard_range_on_memfd() {
        for huge_pages in [HugePageConfig::None, HugePageConfig::Transparent] {
            check_discard_range_on_memfd(huge_pages);
        }
    }

    #[test]
    fn test_discard_range_on_hugetlb_memfd() {
        if free_hugepages_2m() < 3 {
            println!("Skipping: fewer than three free 2 MiB hugepages available");
            return;
        }
        check_discard_range_on_memfd(HugePageConfig::Hugetlbfs2M);
    }

    fn check_discard_range_on_memfd(huge_pages: HugePageConfig) {
        use std::os::unix::fs::FileExt;

        let page_size = huge_pages.page_size();

        // Keep guest addresses distinct from file offsets to exercise offset translation.
        let second_region = GuestAddress((4 * page_size) as u64);
        let discard_addr = second_region.unchecked_add(page_size as u64);
        let regions = [(GuestAddress(0), page_size), (second_region, 2 * page_size)];

        let mem = into_region_ext(
            MemfdBacking::new(3 * page_size as u64, huge_pages)
                .unwrap()
                .allocate(&regions, false, huge_pages)
                .unwrap(),
        );
        mem.write(&vec![1; page_size], GuestAddress(0)).unwrap();
        mem.write(&vec![2; 2 * page_size], second_region).unwrap();

        let region = mem.find_region(second_region).unwrap();
        let file_offset = region.file_offset().unwrap();
        assert_eq!(file_offset.start(), page_size as u64);

        let backing_file = file_offset.file();
        let allocated_before = backing_file.metadata().unwrap().blocks() * 512;
        assert_eq!(allocated_before, (3 * page_size) as u64);

        for (addr, len) in [
            (discard_addr.unchecked_add(1), page_size),
            (discard_addr, page_size - 1),
            (discard_addr.unchecked_add(page_size as u64), page_size),
        ] {
            assert_match!(
                mem.discard_range(addr, len).unwrap_err(),
                GuestMemoryError::InvalidGuestAddress(_)
            );
        }
        assert_eq!(
            backing_file.metadata().unwrap().blocks() * 512,
            allocated_before
        );

        mem.discard_range(discard_addr, page_size).unwrap();
        mem.discard_range(discard_addr, page_size).unwrap();

        // Reading the discarded page can allocate backing again.
        let metadata = backing_file.metadata().unwrap();
        assert_eq!(metadata.blocks() * 512, allocated_before - page_size as u64);
        assert_eq!(metadata.len(), (3 * page_size) as u64);

        let mut page = vec![0; page_size];
        mem.read(&mut page, GuestAddress(0)).unwrap();
        assert_eq!(page, vec![1; page_size]);
        mem.read(&mut page, second_region).unwrap();
        assert_eq!(page, vec![2; page_size]);
        mem.read(&mut page, discard_addr).unwrap();
        assert_eq!(page, vec![0; page_size]);

        mem.write(&vec![3; page_size], discard_addr).unwrap();
        backing_file
            .read_exact_at(&mut page, (2 * page_size) as u64)
            .unwrap();
        assert_eq!(page, vec![3; page_size]);

        if huge_pages == HugePageConfig::Hugetlbfs2M {
            let partial_len = page_size + host_page_size();
            mem.discard_range(second_region, partial_len).unwrap();
            assert_eq!(
                backing_file.metadata().unwrap().blocks() * 512,
                allocated_before - page_size as u64
            );

            let mut pages = vec![0; 2 * page_size];
            mem.read(&mut pages, second_region).unwrap();
            let expected = [vec![0; partial_len], vec![3; 2 * page_size - partial_len]].concat();
            assert_eq!(pages, expected);
        }
    }

    /// Takes the write and discard marks of `mem`, which tracks dirty pages, and runs
    /// `dump_dirty` with them over a file that holds `parent`, as a Diff snapshot merged into
    /// its parent's memory file does. Returns the file's bytes. No KVM dirty log is merged in, as
    /// for a VM whose vCPUs wrote nothing.
    fn merge_diff(mem: &GuestMemoryMmap, parent: &[u8]) -> Vec<u8> {
        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(parent).unwrap();
        file.rewind().unwrap();
        let dirty_bitmap = mem.take_dirty();
        mem.dump_dirty(&mut file, &dirty_bitmap, &mem.take_discarded())
            .unwrap();
        let mut bytes = vec![0u8; parent.len()];
        file.rewind().unwrap();
        file.read_exact(&mut bytes).unwrap();
        bytes
    }

    /// The offset of the first byte where `actual` differs from `expected`, with both bytes.
    fn first_difference(actual: &[u8], expected: &[u8]) -> Option<(usize, u8, u8)> {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
            .map(|offset| (offset, actual[offset], expected[offset]))
    }

    /// Tracked guest memory of `len` bytes in the two kinds a discard frees differently:
    /// anonymous, and mapped privately from a file as after a restore from a snapshot file.
    fn tracked_memories(len: usize) -> [(&'static str, GuestMemoryMmap); 2] {
        let mut backing = TempFile::new().unwrap().into_file();
        backing.write_all(&vec![2u8; len]).unwrap();
        let region = [(GuestAddress(0), len)];
        [
            (
                "anonymous",
                into_region_ext(anonymous(&region, true, HugePageConfig::None).unwrap()),
            ),
            (
                "private file",
                into_region_ext(
                    snapshot_file(backing, &region, true, HugePageConfig::None).unwrap(),
                ),
            ),
        ]
    }

    #[test]
    fn test_diff_writes_discarded_pages_as_zeros_without_reading_them() {
        // A discard empties guest pages without a guest write, so KVM's dirty log never has it.
        // The next Diff has to write zeros over the parent's bytes there, and without reading
        // the pages: a read brings each one back, from the huge page pool on hugetlbfs and from
        // the page server on a UFFD restore.
        let page_size = host_page_size();
        let len = 4 * page_size;
        for (kind, mem) in tracked_memories(len) {
            mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
            mem.reset_dirty();
            // Page 2 is written after the last snapshot, so it carries a dirty mark when it is
            // discarded.
            mem.write(&vec![0x77; page_size], GuestAddress(2 * page_size as u64))
                .unwrap();
            mem.discard_range(GuestAddress(page_size as u64), 2 * page_size)
                .unwrap();

            let mut expected = vec![0xcd; len];
            expected[page_size..3 * page_size].fill(0);
            let diff = merge_diff(&mem, &vec![0xcd; len]);
            assert_eq!(first_difference(&diff, &expected), None, "{kind}: the Diff");
            assert_eq!(
                resident(&mem, GuestAddress(0), len),
                [true, false, false, true],
                "{kind}: the Diff must not read the discarded pages back"
            );
            // The Diff took the marks, so the next one writes nothing.
            let diff = merge_diff(&mem, &vec![0xcd; len]);
            assert_eq!(
                first_difference(&diff, &vec![0xcd; len]),
                None,
                "{kind}: the next Diff"
            );
        }
    }

    #[test]
    fn test_diff_reads_pages_written_after_their_discard() {
        let page_size = host_page_size();
        let len = 3 * page_size;
        for (kind, mem) in tracked_memories(len) {
            mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
            mem.reset_dirty();
            mem.discard_range(GuestAddress(0), 2 * page_size).unwrap();
            mem.write(&vec![0x77; page_size], GuestAddress(page_size as u64))
                .unwrap();

            let mut expected = vec![0xcd; len];
            expected[..page_size].fill(0);
            expected[page_size..2 * page_size].fill(0x77);
            let diff = merge_diff(&mem, &vec![0xcd; len]);
            assert_eq!(first_difference(&diff, &expected), None, "{kind}");
        }
    }

    #[test]
    fn test_diff_reads_a_page_written_after_its_discard_and_swapped_out() {
        // Once the host swaps out a page the guest wrote after its discard, mincore(2) no longer
        // counts it as resident, the same as a page the discard emptied. Its swap entry is what
        // tells the two apart.
        let page_size = host_page_size();
        let len = 2 * page_size;
        let second = GuestAddress(page_size as u64);
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::None).unwrap(),
        );
        mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
        mem.reset_dirty();
        mem.discard_range(second, page_size).unwrap();
        mem.write(&vec![0x66; page_size], second).unwrap();
        let host = mem.get_host_address(second).unwrap();
        let paged_out = unsafe { libc::madvise(host.cast(), page_size, libc::MADV_PAGEOUT) };
        // Reclaim frees the page once its write to swap completes.
        let swapped = paged_out == 0
            && (0..250).any(|_| {
                std::thread::sleep(std::time::Duration::from_millis(20));
                resident(&mem, second, page_size) == [false]
            });
        if !swapped {
            println!(
                "Skipping: the host did not swap the page out (MADV_PAGEOUT returned {paged_out}); \
                 this test needs swap"
            );
            return;
        }

        let mut expected = vec![0xcd; len];
        expected[page_size..].fill(0x66);
        let diff = merge_diff(&mem, &vec![0xcd; len]);
        assert_eq!(first_difference(&diff, &expected), None);
    }

    #[test]
    fn test_diff_after_a_madvise_discard_that_failed_partway() {
        // MADV_DONTNEED drops the pages of one mapping at a time, so a failure on a later mapping
        // leaves the pages before it discarded. Locking the second page makes it fail with EINVAL
        // there, after the first page was dropped. One host page fits the 64 KiB memlock limit
        // of a 64 KiB-page host.
        let page_size = host_page_size();
        let len = 2 * page_size;
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::None).unwrap(),
        );
        mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
        mem.reset_dirty();
        let second = mem
            .get_host_address(GuestAddress(page_size as u64))
            .unwrap();
        let locked = unsafe { libc::mlock(second.cast(), page_size) };
        assert_eq!(locked, 0, "mlock: {}", io::Error::last_os_error());
        assert_match!(
            mem.discard_range(GuestAddress(0), len).unwrap_err(),
            GuestMemoryError::IOError(_)
        );

        let mut expected = vec![0u8; len];
        expected[page_size..].fill(0xab);
        let diff = merge_diff(&mem, &vec![0xcd; len]);
        assert_eq!(first_difference(&diff, &expected), None, "the Diff");
        let mut live = vec![0xffu8; len];
        mem.read(&mut live, GuestAddress(0)).unwrap();
        assert_eq!(first_difference(&live, &expected), None, "guest memory");
        unsafe { libc::munlock(second.cast(), page_size) };
    }

    #[test]
    fn test_diff_writes_the_huge_pages_a_private_hugetlbfs_discard_replaced() {
        const HUGE_PAGE: usize = 2 << 20;
        // Two huge pages hold the memfd and two are reserved for the private copy-on-write.
        let Some(_huge_pages) = test_utils::lock_free_huge_pages(4) else {
            return;
        };
        let backing = test_utils::test_memfd(2 * HUGE_PAGE, true, 0x5a);
        let mem = into_region_ext(
            uffd_minor_backing(
                backing,
                &[(GuestAddress(0), 2 * HUGE_PAGE)],
                true,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        // The range covers the second huge page whole and the first in part, so the second is
        // replaced, and the second half of the first is zeroed in place.
        mem.discard_range(
            GuestAddress((HUGE_PAGE / 2) as u64),
            HUGE_PAGE + HUGE_PAGE / 2,
        )
        .unwrap();

        let mut expected = vec![0xcd; 2 * HUGE_PAGE];
        expected[HUGE_PAGE / 2..].fill(0);
        let diff = merge_diff(&mem, &vec![0xcd; 2 * HUGE_PAGE]);
        assert_eq!(first_difference(&diff, &expected), None, "the Diff");
        assert!(
            !resident(&mem, GuestAddress(HUGE_PAGE as u64), HUGE_PAGE).contains(&true),
            "the Diff must not read the replaced huge page back"
        );
    }

    #[test]
    fn test_diff_writes_the_huge_pages_an_anonymous_hugetlbfs_discard_dropped() {
        const HUGE_PAGE: usize = 2 << 20;
        let Some(_huge_pages) = test_utils::lock_free_huge_pages(3) else {
            return;
        };
        let len = 3 * HUGE_PAGE;
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::Hugetlbfs2M).unwrap(),
        );
        mem.write(&vec![0x5a; len], GuestAddress(0)).unwrap();
        mem.reset_dirty();
        // The range ends halfway into the third huge page, so the second is dropped, and the
        // first half of the third is zeroed in place.
        let discarded = HUGE_PAGE..2 * HUGE_PAGE + HUGE_PAGE / 2;
        mem.discard_range(GuestAddress(HUGE_PAGE as u64), discarded.len())
            .unwrap();

        let mut expected = vec![0xcd; len];
        expected[discarded.clone()].fill(0);
        let diff = merge_diff(&mem, &vec![0xcd; len]);
        assert_eq!(first_difference(&diff, &expected), None, "the Diff");
        assert!(
            !resident(&mem, GuestAddress(HUGE_PAGE as u64), HUGE_PAGE).contains(&true),
            "the Diff must not read the dropped huge page back"
        );
        let mut live = vec![0xffu8; len];
        mem.read(&mut live, GuestAddress(0)).unwrap();
        let mut expected = vec![0x5a; len];
        expected[discarded].fill(0);
        assert_eq!(first_difference(&live, &expected), None, "guest memory");
    }

    #[test]
    fn test_diff_reads_the_partial_huge_page_a_discard_zeroed() {
        // A discard zeroes the part of a huge page it cannot free by writing to it. Neither
        // KVM's dirty log nor a write through the guest memory API records that write, so the
        // discard marks the pages itself.
        const HUGE_PAGE: usize = 2 << 20;
        let Some(_huge_pages) = test_utils::lock_free_huge_pages(1) else {
            return;
        };
        let page_size = host_page_size();
        let mem = into_region_ext(
            anonymous(
                &[(GuestAddress(0), HUGE_PAGE)],
                true,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        mem.write(&vec![0x5a; HUGE_PAGE], GuestAddress(0)).unwrap();
        mem.reset_dirty();
        mem.discard_range(GuestAddress(0), page_size).unwrap();

        let mut expected = vec![0xcd; HUGE_PAGE];
        expected[..page_size].fill(0);
        let diff = merge_diff(&mem, &vec![0xcd; HUGE_PAGE]);
        assert_eq!(first_difference(&diff, &expected), None, "the Diff");
    }

    #[test]
    fn test_diff_after_a_discard_of_shared_memory() {
        // MADV_REMOVE frees the pages of a shared mapping in its file, so the guest reads zeros
        // there afterwards, and the Diff has to hold them.
        let check = |kind: &str, len: usize, huge_pages: HugePageConfig| {
            let mem = into_region_ext(
                MemfdBacking::new(len as u64, huge_pages)
                    .unwrap()
                    .allocate(&[(GuestAddress(0), len)], true, huge_pages)
                    .unwrap(),
            );
            mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
            mem.reset_dirty();
            mem.discard_range(GuestAddress(0), len).unwrap();

            let diff = merge_diff(&mem, &vec![0xcd; len]);
            assert_eq!(
                first_difference(&diff, &vec![0; len]),
                None,
                "{kind}: the Diff"
            );
            assert!(
                !resident(&mem, GuestAddress(0), len).contains(&true),
                "{kind}: the Diff must not read the freed pages back"
            );
            let mut live = vec![0xffu8; len];
            mem.read(&mut live, GuestAddress(0)).unwrap();
            assert!(
                live.iter().all(|&byte| byte == 0),
                "{kind}: the guest reads zeros"
            );
        };
        check("shmem", 2 * host_page_size(), HugePageConfig::None);
        if let Some(_huge_pages) = test_utils::lock_free_huge_pages(1) {
            check("hugetlbfs", 2 << 20, HugePageConfig::Hugetlbfs2M);
        }
    }

    #[test]
    fn test_discard_range_marks_nothing_without_dirty_tracking() {
        // Without dirty tracking the region has no bitmap. The discard succeeds, and a Diff with
        // no dirty page writes nothing.
        let page_size = host_page_size();
        let len = 2 * page_size;
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], false, HugePageConfig::None).unwrap(),
        );
        mem.write(&vec![0xab; len], GuestAddress(0)).unwrap();
        mem.discard_range(GuestAddress(0), page_size).unwrap();
        let mut page = vec![0xffu8; page_size];
        mem.read(&mut page, GuestAddress(0)).unwrap();
        assert!(page.iter().all(|&byte| byte == 0));
        assert!(mem.iter().next().unwrap().bitmap().is_none());
        let discarded = mem.take_discarded();
        assert_eq!(discarded, [vec![0]]);

        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(&vec![0xcd; len]).unwrap();
        file.rewind().unwrap();
        mem.dump_dirty(&mut file, &vec![vec![0]], &discarded)
            .unwrap();
        let mut bytes = vec![0u8; len];
        file.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(first_difference(&bytes, &vec![0xcd; len]), None);
    }

    #[test]
    fn test_unplugged_slot_writes_discarded_pages_as_zeros_without_pagemap() {
        // An unplugged slot is never read, and virtio-mem unplugs a slot before it discards it,
        // so nothing writes a page there after its discard. Every discarded page of the slot is
        // written as zeros, whatever its write marks, and without asking /proc/self/pagemap.
        let page_size = host_page_size();
        let len = 2 * page_size;
        let mem = hotplug_slot_memory(len, false);
        // A write mark left from before the slot was unplugged and discarded.
        mem.mark_dirty(GuestAddress(0), page_size);
        mem.discard_range(GuestAddress(0), len).unwrap();
        // The snapshot writes the page as zeros, so it takes the write mark too.
        assert_eq!(mem.take_dirty(), [vec![0b01]]);
        assert!(!mem.iter().next().unwrap().bitmap().dirty_at(0));
        let discarded = mem.take_discarded();
        assert_eq!(discarded, [vec![0b11]]);

        let mut file = TempFile::new().unwrap().into_file();
        file.write_all(&vec![0xcd; len]).unwrap();
        file.rewind().unwrap();
        let mut scratch = DumpScratch::default();
        let (slot, plugged) = mem.iter().next().unwrap().slots().next().unwrap();
        assert!(!plugged);
        slot.dump_dirty(&mut file, None, &discarded[0], page_size, &mut scratch)
            .unwrap();

        let mut bytes = vec![0xffu8; len];
        file.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(first_difference(&bytes, &vec![0; len]), None);
        assert!(
            scratch.pagemap.is_none(),
            "the dump asked /proc/self/pagemap about an unplugged slot"
        );
    }

    #[test]
    fn test_a_failed_discard_is_remembered_for_an_unplugged_slot_only() {
        // A discard that fails partway (see test_diff_after_a_madvise_discard_that_failed_partway)
        // leaves the range marked written, for the next Diff to read. An unplugged slot cannot
        // be read, so there the failure is remembered instead, until it is told to forget.
        let page_size = host_page_size();
        let len = 2 * page_size;
        for plugged in [true, false] {
            let mem = hotplug_slot_memory(len, plugged);
            let region = mem.iter().next().unwrap();
            let second = mem
                .get_host_address(GuestAddress(page_size as u64))
                .unwrap();
            let locked = unsafe { libc::mlock(second.cast(), page_size) };
            assert_eq!(locked, 0, "mlock: {}", io::Error::last_os_error());

            mem.discard_range(GuestAddress(0), page_size).unwrap();
            assert!(!region.failed_discard_of_unplugged_memory());
            assert_match!(
                mem.discard_range(GuestAddress(0), len).unwrap_err(),
                GuestMemoryError::IOError(_)
            );
            unsafe { libc::munlock(second.cast(), page_size) };

            assert_eq!(region.failed_discard_of_unplugged_memory(), !plugged);
            assert!(region.bitmap().dirty_at(page_size));
            region.forget_failed_discard_of_unplugged_memory();
            assert!(!region.failed_discard_of_unplugged_memory());
        }
    }

    #[test]
    fn test_store_discarded_marks_the_pages_again() {
        let page_size = host_page_size();
        let len = 3 * page_size;
        let mem = into_region_ext(
            anonymous(&[(GuestAddress(0), len)], true, HugePageConfig::None).unwrap(),
        );
        mem.discard_range(GuestAddress(page_size as u64), page_size)
            .unwrap();

        let discarded = mem.take_discarded();
        assert_eq!(discarded, [vec![0b010]]);
        assert_eq!(mem.take_discarded(), [vec![0]], "the take clears the marks");
        mem.store_discarded(&discarded);
        assert_eq!(mem.take_discarded(), [vec![0b010]]);
    }

    #[test]
    fn test_page_action() {
        use PageAction::*;
        // (written, discarded, has a page table entry)
        for (marks, action) in [
            ((false, false, false), Skip),
            ((false, false, true), Skip),
            ((true, false, false), Copy),
            ((true, false, true), Copy),
            ((false, true, false), Zeros),
            ((false, true, true), Zeros),
            // Written and then discarded.
            ((true, true, false), Zeros),
            // Discarded and then written.
            ((true, true, true), Copy),
        ] {
            let (written, discarded, has_entry) = marks;
            assert_eq!(
                page_action(written, discarded, has_entry),
                action,
                "{marks:?}"
            );
        }
    }

    /// A page whose pagemap entry has neither bit 63 nor bit 62 still has a page table entry when
    /// mincore(2) reports it resident, as for a hugetlb migration or hwpoison entry. Such a page,
    /// discarded and then written, is read, not written as zeros. No real entry of that kind is
    /// made here: MADV_HWPOISON poisons a host frame, and a migration window cannot be observed
    /// deterministically.
    #[test]
    fn test_a_page_mincore_reports_resident_has_a_page_table_entry() {
        assert_eq!(
            page_action(true, true, with_entries(&[0], &[1]) == 1),
            PageAction::Copy
        );
        assert_eq!(with_entries(&[0, 0], &[1, 0]), 0b01);
    }

    #[test]
    fn test_a_swapped_pagemap_entry_counts_as_a_page_table_entry() {
        // A small page written after its discard and then swapped out has only pagemap's bit 62,
        // and mincore(2) reports it not resident once it has left the swap cache. It is read, not
        // written as zeros. test_diff_reads_a_page_written_after_its_discard_and_swapped_out
        // covers a real one on a host with swap.
        assert_eq!(
            page_action(true, true, with_entries(&[1 << 62], &[0]) == 1),
            PageAction::Copy
        );
        assert_eq!(with_entries(&[1 << 63, 0, 1 << 62], &[0, 0, 0]), 0b101);
    }

    /// Verifies that `slots_intersecting_range` returns the correct slots for
    /// ranges at slot boundaries, interior to a slot, and spanning two slots.
    #[test]
    fn test_slots_intersecting_range() {
        let page_size = host_page_size();
        let slot_size = 64 * page_size;
        let region_size = 2 * slot_size;
        let base = GuestAddress(0);
        let slot1_base = base.unchecked_add(slot_size as u64);

        let mmap_region = anonymous(&[(base, region_size)], false, HugePageConfig::None)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let region = GuestRegionMmapExt::hotpluggable_from_mmap_region(mmap_region, 0, slot_size);
        assert_eq!(region.slot_cnt(), 2);

        // (range_offset_in_pages, range_len_in_pages, expected_slot_addrs)
        let cases: &[(usize, usize, &[GuestAddress])] = &[
            // At slot 0 boundary
            (0, 1, &[base]),
            // Interior to slot 0
            (1, 1, &[base]),
            // Interior to slot 1
            (65, 1, &[slot1_base]),
            // Spanning slot 0 and slot 1
            (63, 2, &[base, slot1_base]),
            // Entire region
            (0, 128, &[base, slot1_base]),
            // Outside the region
            (128, 1, &[]),
            // Zero-length range
            (0, 0, &[]),
        ];

        for &(offset_pages, len_pages, expected) in cases {
            let from = base.unchecked_add((offset_pages * page_size) as u64);
            let len = len_pages * page_size;
            let found: Vec<_> = region.slots_intersecting_range(from, len).collect();
            let addrs: Vec<_> = found.iter().map(|(s, _)| s.guest_addr).collect();
            assert_eq!(
                addrs, expected,
                "offset={offset_pages} pages, len={len_pages} pages"
            );
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    mod prop_tests {
        use std::io::{Read, Seek, SeekFrom};
        use std::os::unix::fs::MetadataExt;

        use proptest::prelude::*;
        use vmm_sys_util::tempfile::TempFile;

        use super::*;

        /// Naive dump_dirty over a full GuestMemoryMmap: iterates all
        /// regions/slots, seeks over unplugged slots, and for plugged slots
        /// writes the pages set in their region's bitmap one at a time. Returns the number of
        /// dirty pages written.
        fn dump_dirty_oracle(
            mem: &GuestMemoryMmap,
            writer: &mut File,
            dirty_bitmap: &DirtyBitmap,
        ) -> usize {
            let page_size = host_page_size();
            let mut dirty_count = 0;
            for (region, region_bitmap) in mem.iter().zip(dirty_bitmap) {
                let slot_pages = region.slot_size / page_size;
                for (slot_idx, (slot, plugged)) in region.slots().enumerate() {
                    if !plugged {
                        writer
                            .seek(SeekFrom::Current(slot.slice.len() as i64))
                            .unwrap();
                        continue;
                    }

                    for page_index in 0..slot_pages {
                        let bit = slot_idx * slot_pages + page_index;
                        let page_offset = page_index * page_size;
                        if ((region_bitmap[bit / 64] >> (bit % 64)) & 1) != 0 {
                            let slice = &slot.slice.subslice(page_offset, page_size).unwrap();
                            writer.write_all_volatile(slice).unwrap();
                            dirty_count += 1;
                        } else {
                            writer.seek(SeekFrom::Current(page_size as i64)).unwrap();
                        }
                    }
                }
            }
            dirty_count
        }

        /// Generate a KVM dirty bitmap for a slot of `num_pages` pages.
        fn kvm_bitmap_for(num_pages: usize) -> impl Strategy<Value = Vec<u64>> {
            let num_u64s = num_pages.div_ceil(64);
            let last_chunk_valid_bits = num_pages % 64;

            proptest::collection::vec(any::<u64>(), num_u64s).prop_map(move |mut bm| {
                if last_chunk_valid_bits > 0 {
                    let last = bm.len() - 1;
                    bm[last] &= (1u64 << last_chunk_valid_bits) - 1;
                }
                bm
            })
        }

        /// A region descriptor produced by the strategy.
        #[derive(Debug, Clone)]
        struct RegionSpec {
            /// gap (in pages) from the previous region.
            gap_pages: usize,
            /// type of the region
            region_type: GuestRegionType,
            /// size (in pages) of the KVM slots in the region
            pages_per_slot: usize,
            /// array indicating whether each slot is plugged or not
            plugged: Vec<bool>,
            /// mock KVM dirty bitmaps
            /// There is one per slot and each bit of the u64 is a single page
            kvm_bitmaps: Vec<Vec<u64>>,
            /// pages to be accessed by Firecracker during the test.
            /// One bitmap per slot, where each bool is one page
            fc_dirty_pages: Vec<Vec<bool>>,
        }

        /// Strategy for a single region: Dram (1 plugged slot) or
        /// Hotpluggable (1-4 slots, each independently plugged/unplugged). Hotpluggable slots
        /// are a multiple of 64 host pages, so here they are a whole number of bitmap words.
        fn region_spec() -> impl Strategy<Value = RegionSpec> {
            prop_oneof![
                // Dram: 1 slot, always plugged
                (0usize..=8, 1usize..=128).prop_flat_map(|(gap_pages, pages_per_slot)| {
                    (
                        kvm_bitmap_for(pages_per_slot),
                        proptest::collection::vec(any::<bool>(), pages_per_slot),
                    )
                        .prop_map(move |(bm, fc)| RegionSpec {
                            gap_pages,
                            region_type: GuestRegionType::Dram,
                            pages_per_slot,
                            plugged: vec![true],
                            kvm_bitmaps: vec![bm],
                            fc_dirty_pages: vec![fc],
                        })
                }),
                // Hotpluggable: 1-4 slots, each plugged or not
                (0usize..=8, 1usize..=2, 1usize..=4).prop_flat_map(
                    |(gap_pages, words_per_slot, num_slots)| {
                        let pages_per_slot = words_per_slot * 64;
                        (
                            proptest::collection::vec(any::<bool>(), num_slots),
                            proptest::collection::vec(kvm_bitmap_for(pages_per_slot), num_slots),
                            proptest::collection::vec(
                                proptest::collection::vec(any::<bool>(), pages_per_slot),
                                num_slots,
                            ),
                        )
                            .prop_map(
                                move |(plugged, kvm_bitmaps, fc_dirty_pages)| RegionSpec {
                                    gap_pages,
                                    region_type: GuestRegionType::Hotpluggable,
                                    pages_per_slot,
                                    plugged,
                                    kvm_bitmaps,
                                    fc_dirty_pages,
                                },
                            )
                    },
                ),
            ]
        }

        /// Build a GuestMemoryMmap and the KVM dirty log of each plugged slot from region specs.
        fn build_memory(specs: &[RegionSpec]) -> (GuestMemoryMmap, HashMap<u32, Vec<u64>>, usize) {
            let page_size = host_page_size();
            let mut slot_from = 0u32;
            let mut regions = Vec::new();
            let mut kvm_bitmap = HashMap::new();
            let mut total_size = 0usize;
            let mut next_addr = 0u64;

            for spec in specs {
                next_addr += (spec.gap_pages * page_size) as u64;
                let num_slots = spec.plugged.len();
                let region_size = num_slots * spec.pages_per_slot * page_size;

                let mmap_regions = anonymous(
                    &[(GuestAddress(next_addr), region_size)],
                    true,
                    HugePageConfig::None,
                )
                .unwrap();

                let state = GuestMemoryRegionState {
                    base_address: next_addr,
                    size: region_size,
                    region_type: spec.region_type,
                    plugged: spec.plugged.clone(),
                };

                let region = GuestRegionMmapExt::from_state(
                    mmap_regions.into_iter().next().unwrap(),
                    &state,
                    slot_from,
                )
                .unwrap();

                // KVM only logs plugged slots.
                for (i, bm) in spec.kvm_bitmaps.iter().enumerate() {
                    if spec.plugged[i] {
                        kvm_bitmap.insert(slot_from + i as u32, bm.clone());
                    }
                }

                regions.push(region);
                slot_from += num_slots as u32;
                total_size += region_size;
                next_addr += region_size as u64;
            }

            (
                GuestMemoryMmap::from_regions(regions).unwrap(),
                kvm_bitmap,
                total_size,
            )
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(4096))]

            #[test]
            fn dump_dirty_correctness(
                region_specs in proptest::collection::vec(region_spec(), 1..=3),
            ) {
                let page_size = host_page_size();
                let (guest_memory, kvm_bitmap, total_size) =
                    build_memory(&region_specs);

                // Fill backing memory with non-zero data via raw pointer so
                // that KVM-only-dirty pages carry distinguishable content
                // without triggering the firecracker bitmap.
                for region in guest_memory.iter() {
                    let ptr = region
                        .get_host_address(MemoryRegionAddress(0))
                        .unwrap();
                    // SAFETY: ptr is valid for region.len() bytes.
                    unsafe { std::ptr::write_bytes(ptr, 0xAB, u64_to_usize(region.len())) };
                }

                // Dirty selected pages in the firecracker bitmap.
                for (region, spec) in guest_memory.iter().zip(region_specs.iter()) {
                    for (slot_idx, (slot, plugged)) in region.slots().enumerate() {
                        if !plugged {
                            continue;
                        }
                        for (page, dirty) in spec.fc_dirty_pages[slot_idx].iter().enumerate() {
                            if *dirty {
                                let addr = slot.guest_addr.0 + (page * page_size) as u64;
                                guest_memory.write(&[0xCD], GuestAddress(addr)).unwrap();
                            }
                        }
                    }
                }

                let mut dirty_bitmap = guest_memory.take_dirty();
                guest_memory
                    .merge_slot_kvm_dirty_logs(&mut dirty_bitmap, |slot| {
                        Ok::<_, ()>(kvm_bitmap[&slot.slot].clone())
                    })
                    .unwrap();

                let mut oracle_file = TempFile::new().unwrap().into_file();
                oracle_file.set_len(total_size as u64).unwrap();
                let dirty_count = dump_dirty_oracle(&guest_memory, &mut oracle_file, &dirty_bitmap);
                let expected_blocks = (dirty_count * page_size) as u64 / 512;
                let oracle_pos = oracle_file.stream_position().unwrap();

                // sanity check the oracle implementation
                prop_assert_eq!(oracle_pos, total_size as u64);
                prop_assert_eq!(oracle_file.metadata().unwrap().blocks(), expected_blocks);

                // Run the optimized implementation.
                let mut opt_file = TempFile::new().unwrap().into_file();
                opt_file.set_len(total_size as u64).unwrap();
                guest_memory
                    .dump_dirty(&mut opt_file, &dirty_bitmap, &guest_memory.take_discarded())
                    .unwrap();
                let opt_pos = opt_file.stream_position().unwrap();

                // check the writer actually moved the cursor to the end and wrote all dirty blocks
                prop_assert_eq!(opt_pos, total_size as u64);
                prop_assert_eq!(opt_file.metadata().unwrap().blocks(), expected_blocks);

                // Read back and compare file contents.
                opt_file.seek(SeekFrom::Start(0)).unwrap();
                oracle_file.seek(SeekFrom::Start(0)).unwrap();
                let mut opt_buf = vec![0u8; total_size];
                let mut oracle_buf = vec![0u8; total_size];
                opt_file.read_exact(&mut opt_buf).unwrap();
                oracle_file.read_exact(&mut oracle_buf).unwrap();
                prop_assert_eq!(&opt_buf, &oracle_buf);
            }

            #[test]
            fn take_dirty_correctness(
                region_specs in proptest::collection::vec(region_spec(), 1..=3),
            ) {
                let page_size = host_page_size();
                let (guest_memory, kvm_bitmap, _) = build_memory(&region_specs);

                // Mark pages in the firecracker bitmap, in unplugged slots too.
                for (region, spec) in guest_memory.iter().zip(region_specs.iter()) {
                    for (slot_idx, (slot, _)) in region.slots().enumerate() {
                        for (page, dirty) in spec.fc_dirty_pages[slot_idx].iter().enumerate() {
                            if *dirty {
                                let addr = slot.guest_addr.0 + (page * page_size) as u64;
                                guest_memory.mark_dirty(GuestAddress(addr), 1);
                            }
                        }
                    }
                }

                let mut dirty_bitmap = guest_memory.take_dirty();
                guest_memory
                    .merge_slot_kvm_dirty_logs(&mut dirty_bitmap, |slot| {
                        Ok::<_, ()>(kvm_bitmap[&slot.slot].clone())
                    })
                    .unwrap();

                // Every region holds its plugged slots' KVM bits OR its firecracker bits, and
                // nothing past its last page; the firecracker bitmap is clear.
                prop_assert_eq!(dirty_bitmap.len(), region_specs.len());
                for ((region, spec), taken) in
                    guest_memory.iter().zip(region_specs.iter()).zip(&dirty_bitmap)
                {
                    let num_pages = region.size() / page_size;
                    prop_assert_eq!(taken.len(), num_pages.div_ceil(64));
                    for bit in 0..taken.len() * 64 {
                        let (slot_idx, page) = (bit / spec.pages_per_slot, bit % spec.pages_per_slot);
                        let in_region = bit < num_pages;
                        let kvm_dirty = in_region
                            && spec.plugged[slot_idx]
                            && ((spec.kvm_bitmaps[slot_idx][page / 64] >> (page % 64)) & 1) == 1;
                        let fc_dirty = in_region && spec.fc_dirty_pages[slot_idx][page];
                        let is_taken = ((taken[bit / 64] >> (bit % 64)) & 1) == 1;
                        prop_assert_eq!(is_taken, kvm_dirty || fc_dirty, "mismatch at bit {}", bit);
                    }
                    for (slot, _) in region.slots() {
                        for page in 0..slot.slice.len() / page_size {
                            prop_assert!(!slot.slice.bitmap().dirty_at(page * page_size));
                        }
                    }
                }
            }

            #[test]
            fn store_dirty_bitmap_correctness(
                region_specs in proptest::collection::vec(region_spec(), 1..=3),
            ) {
                let page_size = host_page_size();
                let (guest_memory, _, _) = build_memory(&region_specs);

                // Every other region gets an empty bitmap. Hotpluggable slots are whole words, so a
                // region's bitmap is its slots' bitmaps one after another, unplugged ones included.
                let dirty_bitmap: DirtyBitmap = region_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| match i % 2 {
                        0 => spec.kvm_bitmaps.concat(),
                        _ => Vec::new(),
                    })
                    .collect();

                guest_memory.store_dirty_bitmap(&dirty_bitmap);

                // The firecracker bitmap starts clean, so it must now match `dirty_bitmap` exactly.
                for (region, region_bitmap) in guest_memory.iter().zip(&dirty_bitmap) {
                    let slot_pages = region.slot_size / page_size;
                    for (slot_idx, (slot, _)) in region.slots().enumerate() {
                        let fc = slot.slice.bitmap();
                        for page in 0..slot_pages {
                            let bit = slot_idx * slot_pages + page;
                            let stored = region_bitmap
                                .get(bit / 64)
                                .is_some_and(|word| ((word >> (bit % 64)) & 1) == 1);
                            prop_assert_eq!(
                                fc.dirty_at(page * page_size), stored, "mismatch at page {}", page
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_check_range_plugged() {
        let region_size = 0x4000usize; // 4 slots of 0x1000
        let regions = anonymous(
            &[(GuestAddress(0x10_0000), region_size)],
            false,
            HugePageConfig::None,
        )
        .unwrap();
        let region = regions.into_iter().next().unwrap();

        let state = GuestMemoryRegionState {
            base_address: 0x10_0000,
            size: region_size,
            region_type: GuestRegionType::Hotpluggable,
            plugged: vec![true, true, false, true],
        };

        let ext = GuestRegionMmapExt::from_state(region, &state, 0).unwrap();

        // Slot 0 (offset 0..0x1000): plugged
        ext.check_range_plugged(MemoryRegionAddress(0), 0x100)
            .unwrap();
        // Slot 1 (offset 0x1000..0x2000): plugged
        ext.check_range_plugged(MemoryRegionAddress(0x1000), 0x100)
            .unwrap();
        // Slot 2 (offset 0x2000..0x3000): unplugged
        assert!(
            ext.check_range_plugged(MemoryRegionAddress(0x2000), 0x100)
                .is_err()
        );
        // Spanning slots 1-2: fails because slot 2 is unplugged
        assert!(
            ext.check_range_plugged(MemoryRegionAddress(0x1800), 0x1000)
                .is_err()
        );
        // Spanning slots 0-1: both plugged
        ext.check_range_plugged(MemoryRegionAddress(0x800), 0x1000)
            .unwrap();
        // Slot 3 (offset 0x3000..0x4000): plugged
        ext.check_range_plugged(MemoryRegionAddress(0x3000), 0x100)
            .unwrap();
        // Zero length: always ok
        ext.check_range_plugged(MemoryRegionAddress(0x2000), 0)
            .unwrap();
    }

    /// Returns the (start, end) of the VMA containing `addr` in `/proc/self/maps`, if any.
    fn find_vma_containing(addr: usize) -> Option<(usize, usize)> {
        use std::io::BufRead;
        let maps = std::fs::File::open("/proc/self/maps").unwrap();
        for line in std::io::BufReader::new(maps).lines() {
            let line = line.unwrap();
            let range = line.split_whitespace().next().unwrap_or("");
            let mut parts = range.split('-');
            let start = usize::from_str_radix(parts.next().unwrap_or("0"), 16).unwrap_or(0);
            let end = usize::from_str_radix(parts.next().unwrap_or("0"), 16).unwrap_or(0);
            if addr >= start && addr < end {
                return Some((start, end));
            }
        }
        None
    }

    fn vma_has_vm_flag(addr: usize, expected_flag: &str) -> bool {
        use std::io::BufRead;

        let smaps = std::fs::File::open("/proc/self/smaps").unwrap();
        let mut in_target_vma = false;
        for line in std::io::BufReader::new(smaps).lines() {
            let line = line.unwrap();
            if let Some(range) = line.split_whitespace().next()
                && let Some((start, end)) = range.split_once('-')
                && let (Ok(start), Ok(end)) = (
                    usize::from_str_radix(start, 16),
                    usize::from_str_radix(end, 16),
                )
            {
                in_target_vma = addr >= start && addr < end;
                continue;
            }
            if in_target_vma && line.starts_with("VmFlags:") {
                return line
                    .split_whitespace()
                    .skip(1)
                    .any(|flag| flag == expected_flag);
            }
        }
        panic!("mapping containing address {addr:#x} has no VmFlags entry")
    }

    #[test]
    fn test_lock_free_huge_pages_skips_when_the_pool_is_short() {
        // The skip path must be reachable, or a short pool would fail the huge page tests.
        assert!(test_utils::lock_free_huge_pages(u64::MAX).is_none());
    }

    /// The `resv_hugepages` counter of the 2 MiB pool.
    fn reserved_hugepages_2m() -> u64 {
        std::fs::read_to_string("/sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages")
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    /// A UFFD minor backing on shmem is mapped `MAP_NORESERVE` like other guest memory. One on
    /// hugetlbfs is not: the kernel reserves a huge page for the copy of each one mapped, so a
    /// short pool fails the restore instead of a guest write.
    #[test]
    fn test_uffd_minor_backing_reserves_copies_on_hugetlbfs_only() {
        const HUGE_PAGE: usize = 2 << 20;
        let regions = [(GuestAddress(0), 2 * HUGE_PAGE)];

        let shmem = test_utils::test_memfd(2 * HUGE_PAGE, false, 0x5a);
        let mapped = uffd_minor_backing(shmem, &regions, false, HugePageConfig::None).unwrap();
        assert!(vma_has_vm_flag(mapped[0].as_ptr() as usize, "nr"));
        assert_eq!(mapped[0].flags() & libc::MAP_PRIVATE, libc::MAP_PRIVATE);
        drop(mapped);

        // Two pages for the file and two reserved for their copies.
        let Some(_pool) = test_utils::lock_free_huge_pages(4) else {
            return;
        };
        let hugetlbfs = test_utils::test_memfd(2 * HUGE_PAGE, true, 0x5a);
        let reserved_before = reserved_hugepages_2m();
        let mapped =
            uffd_minor_backing(hugetlbfs, &regions, false, HugePageConfig::Hugetlbfs2M).unwrap();
        assert!(!vma_has_vm_flag(mapped[0].as_ptr() as usize, "nr"));
        assert_eq!(reserved_hugepages_2m(), reserved_before + 2);
        // The region knows its backing page size, which is what a discard aligns to.
        assert_eq!(RawGuestRegionMmap::page_size(mapped[0].flags()), HUGE_PAGE);
        drop(mapped);
        assert_eq!(reserved_hugepages_2m(), reserved_before);
    }

    /// A discard on a UFFD minor restore with 2 MiB pages, a private mapping of a hugetlbfs
    /// file: the huge pages the range covers whole are replaced with anonymous memory, the
    /// partly covered ones at either end are zeroed where the range covers them, and the
    /// handler's file keeps its bytes. The kernel cannot split a hugetlbfs mapping inside a
    /// huge page, so a replacement that was not aligned to the backing page size would fail
    /// after `MAP_FIXED` had unmapped the range.
    #[test]
    fn test_discard_range_on_a_uffd_minor_hugetlbfs_backing() {
        const HUGE_PAGE: usize = 2 << 20;
        // Three pages for the file and three reserved for their copies.
        let Some(_pool) = test_utils::lock_free_huge_pages(6) else {
            return;
        };
        let file = test_utils::test_memfd(3 * HUGE_PAGE, true, 0x5a);
        let mem = into_region_ext(
            uffd_minor_backing(
                file.try_clone().unwrap(),
                &[(GuestAddress(0), 3 * HUGE_PAGE)],
                false,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );

        // From the middle of the first huge page to the middle of the third.
        let start = HUGE_PAGE / 2;
        let len = 2 * HUGE_PAGE;
        mem.discard_range(GuestAddress(start as u64), len).unwrap();

        let mut actual = vec![0xff; 3 * HUGE_PAGE];
        mem.read(&mut actual, GuestAddress(0)).unwrap();
        let expected = [
            vec![0x5a; start],
            vec![0; len],
            vec![0x5a; 3 * HUGE_PAGE - start - len],
        ]
        .concat();
        assert!(actual == expected, "guest memory after the discard");

        let mut backing = vec![0; 3 * HUGE_PAGE];
        file.read_exact_at(&mut backing, 0).unwrap();
        assert!(
            backing.iter().all(|byte| *byte == 0x5a),
            "the discard wrote to the handler's backing file"
        );
    }

    #[test]
    fn test_which_pagemap_entries_a_write_would_allocate_for() {
        const PRESENT: u64 = 1 << 63;
        const FILE_PAGE: u64 = 1 << 61;
        // Not in memory: any write allocates.
        assert!(pagemap_entry_write_allocates(0, false));
        assert!(pagemap_entry_write_allocates(0, true));
        // The mapping's own page.
        assert!(!pagemap_entry_write_allocates(PRESENT, false));
        // A page of the file: a private mapping copies it, a shared one writes it in place.
        assert!(pagemap_entry_write_allocates(PRESENT | FILE_PAGE, false));
        assert!(!pagemap_entry_write_allocates(PRESENT | FILE_PAGE, true));
    }

    /// Free 2 MiB pages of the host, for tests that hold [`test_utils::lock_free_huge_pages`].
    fn free_huge_pages() -> u64 {
        std::fs::read_to_string("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages")
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    /// A clone of a UFFD minor-fault handler on 2 MiB pages maps the handler's file privately.
    /// A balloon range from the middle of the first huge page to the middle of the third
    /// covers the second whole and the other two in part. The clone has written none of them,
    /// so zeroing the two parts would copy two huge pages for it: the range must cost nothing.
    #[test]
    fn test_a_ballooned_range_allocates_no_huge_page_for_a_page_it_covers_in_part() {
        const HUGE_PAGE: usize = 2 << 20;
        let Some(_pool) = test_utils::lock_free_huge_pages(6) else {
            return;
        };
        let file = test_utils::test_memfd(3 * HUGE_PAGE, true, 0x5a);
        let mem = into_region_ext(
            uffd_minor_backing(
                file.try_clone().unwrap(),
                &[(GuestAddress(0), 3 * HUGE_PAGE)],
                false,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        let start = HUGE_PAGE / 2;
        let len = 2 * HUGE_PAGE;

        let free_before = free_huge_pages();
        mem.discard_ballooned_range(GuestAddress(start as u64), len)
            .unwrap();
        assert_eq!(
            free_huge_pages(),
            free_before,
            "the balloon discard allocated huge pages"
        );

        // The second huge page was freed and reads as zeros. The parts of the first and third
        // keep the file's bytes, which are this guest's own from before the snapshot.
        let mut actual = vec![0xff; 3 * HUGE_PAGE];
        mem.read(&mut actual, GuestAddress(0)).unwrap();
        let expected = [
            vec![0x5a; HUGE_PAGE],
            vec![0; HUGE_PAGE],
            vec![0x5a; HUGE_PAGE],
        ]
        .concat();
        assert!(actual == expected, "guest memory after the balloon discard");
        let mut backing = vec![0; 3 * HUGE_PAGE];
        file.read_exact_at(&mut backing, 0).unwrap();
        assert!(
            backing.iter().all(|byte| *byte == 0x5a),
            "the discard wrote to the handler's backing file"
        );
    }

    /// The same range after the clone has written all three huge pages, so each is a private
    /// copy: the parts in the range are zeroed in place, which allocates nothing, and the
    /// huge page the range covers whole is freed.
    #[test]
    fn test_a_ballooned_range_zeroes_private_huge_pages_and_frees_whole_ones() {
        const HUGE_PAGE: usize = 2 << 20;
        let Some(_pool) = test_utils::lock_free_huge_pages(6) else {
            return;
        };
        let file = test_utils::test_memfd(3 * HUGE_PAGE, true, 0x5a);
        let mem = into_region_ext(
            uffd_minor_backing(
                file.try_clone().unwrap(),
                &[(GuestAddress(0), 3 * HUGE_PAGE)],
                false,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        let start = HUGE_PAGE / 2;
        let len = 2 * HUGE_PAGE;

        let free_untouched = free_huge_pages();
        for page in 0..3 {
            mem.write_obj(0x77u8, GuestAddress((page * HUGE_PAGE) as u64))
                .unwrap();
        }
        assert_eq!(
            free_huge_pages(),
            free_untouched - 3,
            "three private copies"
        );

        mem.discard_ballooned_range(GuestAddress(start as u64), len)
            .unwrap();
        assert_eq!(
            free_huge_pages(),
            free_untouched - 2,
            "the whole huge page was not freed, or a part cost a page"
        );

        let mut actual = vec![0xff; 3 * HUGE_PAGE];
        mem.read(&mut actual, GuestAddress(0)).unwrap();
        let mut expected = [
            vec![0x5a; start],
            vec![0; len],
            vec![0x5a; 3 * HUGE_PAGE - start - len],
        ]
        .concat();
        expected[0] = 0x77;
        assert!(actual == expected, "guest memory after the balloon discard");
    }

    /// Anonymous 2 MiB pages, as a VM booted with huge pages has: a balloon range over parts of
    /// two huge pages nothing has touched allocates neither, and they read as zeros, which is
    /// what the zeroing would have left.
    #[test]
    fn test_a_ballooned_range_allocates_no_anonymous_huge_page() {
        const HUGE_PAGE: usize = 2 << 20;
        let Some(_pool) = test_utils::lock_free_huge_pages(4) else {
            return;
        };
        let mem = into_region_ext(
            anonymous(
                &[(GuestAddress(0), 2 * HUGE_PAGE)],
                false,
                HugePageConfig::Hugetlbfs2M,
            )
            .unwrap(),
        );
        let free_before = free_huge_pages();
        mem.discard_ballooned_range(GuestAddress((HUGE_PAGE / 2) as u64), HUGE_PAGE)
            .unwrap();
        assert_eq!(free_huge_pages(), free_before);

        // Written, a part is the mapping's own and is zeroed.
        mem.write_slice(&[0x33; 8192], GuestAddress((HUGE_PAGE / 2) as u64))
            .unwrap();
        mem.discard_ballooned_range(GuestAddress((HUGE_PAGE / 2) as u64), HUGE_PAGE)
            .unwrap();
        let mut actual = vec![0xff; 2 * HUGE_PAGE];
        mem.read(&mut actual, GuestAddress(0)).unwrap();
        assert!(actual.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn test_guest_region_errors_when_empty_size() {
        assert!(matches!(
            RawGuestRegionMmap::allocate_protected(0, 0),
            Err(MemoryError::ZeroSize)
        ));
    }

    #[test]
    fn test_extract_huge_tlb_mmap_flags() {
        let cases = [
            (0, 0),
            (
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                0,
            ),
            (libc::MAP_HUGETLB, libc::MAP_HUGETLB),
            (
                libc::MAP_PRIVATE
                    | libc::MAP_ANONYMOUS
                    | libc::MAP_NORESERVE
                    | libc::MAP_HUGETLB
                    | libc::MAP_HUGE_2MB,
                libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            ),
            (
                libc::MAP_SHARED | libc::MAP_HUGETLB | libc::MAP_HUGE_1GB,
                libc::MAP_HUGETLB | libc::MAP_HUGE_1GB,
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(
                RawGuestRegionMmap::extract_huge_tlb_mmap_flags(input),
                expected,
                "input={input:#x}"
            );
        }
    }

    /// Returns the number of free 2 MiB hugepages on the system.
    fn free_hugepages_2m() -> usize {
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        meminfo
            .lines()
            .find(|line| line.starts_with("HugePages_Free:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    }

    #[test]
    fn test_guest_region_alignment_and_size_normal() {
        // Note: this covers both the huge_pages = None and THP cases. That's because both allocate the underlying
        // memory using the host page size (e.g. 4KiB).
        test_guest_region_alignment_and_size(host_page_size(), 0);
    }

    #[test]
    fn test_guest_region_alignment_and_size_huge() {
        if free_hugepages_2m() == 0 {
            println!("Skipping: no free 2 MiB hugepages available");
            return;
        }
        test_guest_region_alignment_and_size(
            mib_to_bytes(2),
            libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
        );
    }

    fn test_guest_region_alignment_and_size(page_size: usize, mmap_hugetlb_flags: libc::c_int) {
        let expected_alignment = GUEST_MEMORY_ALIGNMENT;

        // Base sizes: powers of two from 1 KiB to 16 MiB.
        let base_sizes: Vec<usize> = (10..=24).map(|shift| 1usize << shift).collect();

        // Offsets to apply to each base size.
        // The idea is to test memory allocation for sizes that aren't necessarily page multiples.
        let offsets: &[isize] = &[-1, 0, 1, 1023, 1024, 1025];

        for base in base_sizes {
            for offset in offsets {
                let size = base.checked_add_signed(*offset).unwrap();

                let region =
                    RawGuestRegionMmap::allocate_protected(size, mmap_hugetlb_flags).unwrap();
                let addr = region.mmap_address as usize;

                // Returned address is 2MB-aligned.
                assert_eq!(
                    addr % expected_alignment,
                    0,
                    "Address not 2MB-aligned for base={base:#x} offset={offset}"
                );

                // Returned size is the requested size rounded up to page alignment.
                assert!(
                    region.mmap_size >= size && region.mmap_size < size + page_size,
                    "mmap_size {mmap_size} unexpected for size={size} (base={base:#x} \
                     offset={offset})",
                    mmap_size = region.mmap_size
                );
                assert_eq!(
                    region.mmap_size % page_size,
                    0,
                    "mmap_size not page-aligned for base={base:#x} offset={offset}"
                );

                // Verify the VMA spans exactly the page-aligned size (head/tail unmapped).
                let (vma_start, vma_end) = find_vma_containing(addr).unwrap_or_else(|| {
                    panic!(
                        "Aligned region not in /proc/self/maps for base={base:#x} offset={offset}"
                    )
                });
                let vma_size = vma_end - vma_start;
                // Note: the kernel might merge our VMA with a neighbor, we can only check that our region is
                // contained within the (larger) merged VMA.
                assert!(
                    vma_start <= addr,
                    "VMA start mismatch for base={base:#x} offset={offset}"
                );
                assert!(
                    vma_size >= region.mmap_size,
                    "VMA size mismatch for base={base:#x} offset={offset}"
                );
            }
        }
    }

    #[test]
    fn test_guest_region_drop_reclaims_memory() {
        let size = mib_to_bytes(4);
        let region = RawGuestRegionMmap::allocate_protected(size, 0).unwrap();
        let addr = region.mmap_address as usize;

        // The region should be mapped before drop.
        assert!(find_vma_containing(addr).is_some());

        // Drop the region.
        drop(region);

        // After drop, the memory should no longer be mapped.
        assert!(
            find_vma_containing(addr).is_none(),
            "Memory was not reclaimed after drop"
        );
    }

    /// The policy `/proc/self/numa_maps` reports for the mapping that holds `addr`.
    fn numa_policy_at(addr: usize) -> String {
        // The kernel merges adjacent anonymous mappings with one policy, so find the VMA that
        // holds `addr` in maps, then its line in numa_maps.
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let start = maps
            .lines()
            .filter_map(|line| {
                let (lo, rest) = line.split_once('-')?;
                let hi = rest.split_whitespace().next()?;
                let lo = usize::from_str_radix(lo, 16).ok()?;
                let hi = usize::from_str_radix(hi, 16).ok()?;
                (lo <= addr && addr < hi).then_some(lo)
            })
            .next()
            .unwrap_or_else(|| panic!("no mapping holds {addr:x}"));
        let numa_maps = std::fs::read_to_string("/proc/self/numa_maps").unwrap();
        let prefix = format!("{start:x} ");
        numa_maps
            .lines()
            .find(|line| line.starts_with(&prefix))
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_else(|| panic!("no mapping at {start:x} in numa_maps"))
            .to_string()
    }

    /// Guest memory is bound MPOL_LOCAL (out of NUMA balancing) for the window, then returned
    /// to the default policy.
    #[test]
    fn guest_memory_leaves_numa_balancing_for_the_window_then_rejoins() {
        if !std::path::Path::new("/proc/self/numa_maps").exists() {
            // A kernel without NUMA has no balancing to stay out of.
            return;
        }
        let memory = anonymous(
            &[(GuestAddress(0), 4 << 20), (GuestAddress(8 << 20), 4 << 20)],
            false,
            HugePageConfig::None,
        )
        .unwrap();
        let memory = GuestMemoryMmap::from_regions(
            memory
                .into_iter()
                .map(|region| GuestRegionMmapExt::dram_from_mmap_region(region, 0))
                .collect(),
        )
        .unwrap();
        let addrs: Vec<usize> = memory.iter().map(|r| r.as_ptr() as usize).collect();
        for &addr in &addrs {
            assert_eq!(numa_policy_at(addr), "default");
        }
        defer_numa_balancing_for(&memory, 1, Arc::new(vec![]));
        for &addr in &addrs {
            assert_eq!(numa_policy_at(addr), "local", "the window did not start");
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
        for &addr in &addrs {
            assert_eq!(numa_policy_at(addr), "default", "the window did not end");
        }
        // 0 leaves the memory under the default policy.
        defer_numa_balancing_for(&memory, 0, Arc::new(vec![]));
        assert_eq!(numa_policy_at(addrs[0]), "default");
    }

    /// `len` bytes of `backing` mapped as a restore maps a snapshot's memory: one region,
    /// `MAP_PRIVATE`.
    fn restored_from(backing: &File, len: usize) -> GuestMemoryMmap {
        let regions = snapshot_file(
            backing.try_clone().unwrap(),
            &[(GuestAddress(0), len)],
            false,
            HugePageConfig::None,
        )
        .unwrap();
        GuestMemoryMmap::from_regions(
            regions
                .into_iter()
                .map(|region| GuestRegionMmapExt::dram_from_mmap_region(region, 0))
                .collect(),
        )
        .unwrap()
    }

    /// The clones of one UFFD minor-fault handler map one memfd, and a shmem mapping keeps its
    /// NUMA policy in the file, where every mapping of it reads it. One clone's window must
    /// not change another clone's policy, when it starts or when it ends, so memory that maps
    /// a memfd is not bound.
    #[test]
    fn one_clone_s_numa_window_leaves_another_mapping_of_the_memfd_alone() {
        if !std::path::Path::new("/proc/self/numa_maps").exists() {
            return;
        }
        let len = 4 << 20;
        let backing = test_utils::test_memfd(len, false, 0x5a);
        let first = restored_from(&backing, len);
        let second = restored_from(&backing, len);
        let addr = |memory: &GuestMemoryMmap| memory.iter().next().unwrap().as_ptr() as usize;
        assert_eq!(numa_policy_at(addr(&first)), "default");
        assert_eq!(numa_policy_at(addr(&second)), "default");
        defer_numa_balancing_for(&first, 1, Arc::new(vec![]));
        assert_eq!(
            numa_policy_at(addr(&second)),
            "default",
            "one clone's window changed the policy of another clone's guest memory"
        );
        assert_eq!(numa_policy_at(addr(&first)), "default");
        // Past the end of the window the first clone asked for.
        std::thread::sleep(std::time::Duration::from_millis(1500));
        assert_eq!(numa_policy_at(addr(&second)), "default");
        assert_eq!(numa_policy_at(addr(&first)), "default");
    }

    /// A private mapping of an ordinary file, which is what a restore from the snapshot's memory
    /// file is, has a policy of its own and is bound for the window.
    #[test]
    fn a_private_mapping_of_an_ordinary_file_is_bound_for_the_window() {
        if !std::path::Path::new("/proc/self/numa_maps").exists() {
            return;
        }
        let len = 4 << 20;
        let backing = TempFile::new().unwrap().into_file();
        backing.set_len(len as u64).unwrap();
        // Where the temporary directory is a tmpfs the file is shmem, and is left alone.
        let private = !BackingFileSystem::of(&backing).unwrap().is_shmem();
        let memory = restored_from(&backing, len);
        let addr = memory.iter().next().unwrap().as_ptr() as usize;
        assert_eq!(numa_policy_at(addr), "default");
        defer_numa_balancing_for(&memory, 1, Arc::new(vec![]));
        assert_eq!(
            numa_policy_at(addr),
            if private { "local" } else { "default" }
        );
        std::thread::sleep(std::time::Duration::from_millis(1500));
        assert_eq!(numa_policy_at(addr), "default");
    }
}

#[cfg(kani)]
mod verification {
    use std::num::NonZeroUsize;

    use super::*;

    fn is_set(bits: &[u64], bit: usize) -> bool {
        bits[bit / 64] & (1 << (bit % 64)) != 0
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn verify_set_bits() {
        const PAGES: usize = 128;
        let bitmap = AtomicBitmap::new(PAGES, NonZeroUsize::MIN);
        let len = kani::any_where(|len: &usize| (1..=PAGES).contains(len));
        // A bitmap never has bits past its last page. Where its bits are is arbitrary, but
        // at most two per word keep the proof small: `set_bits` sets each bit independently.
        let mut bits: [u64; 2] = kani::any();
        for (word, word_bits) in bits.iter_mut().enumerate() {
            let valid = len.saturating_sub(word * 64).min(64);
            *word_bits &= u64::MAX.checked_shr(64 - valid as u32).unwrap_or(0);
            kani::assume(word_bits.count_ones() <= 2);
        }
        // `set_bits` must also keep a page that was marked before it ran.
        let page = kani::any_where(|page: &usize| *page < PAGES);
        let marked: bool = kani::any();
        if marked {
            bitmap.set_bit(page);
        }

        set_bits(&bitmap, &bits[..len.div_ceil(64)]);

        let in_bits = page < len && is_set(&bits, page);
        assert_eq!(bitmap.is_bit_set(page), marked || in_bits);
        kani::cover!(page >= 64 && in_bits);
    }
}
