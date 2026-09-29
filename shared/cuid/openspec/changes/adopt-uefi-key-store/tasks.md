## 1. efivarfs

- [ ] 1.1 A table of {GUID, name} secret variables beside
      `variable_validate[]`; inodes created 0600 at population and on create.
      First entry `LoaderSystemToken`.
- [ ] 1.2 `AMD_CUID_EFI_GUID` in `include/linux/efi.h` and the `AmdCuidKey`
      entry.

## 2. Kernel (amdgpu)

- [x] 2.1 Load or create `AmdCuidKey` once per module lifetime, outside
      `cuid_seed_lock`, guarded by `IS_ENABLED(CONFIG_EFI)`; `MODULE_IMPORT_NS`.
- [x] 2.2 Publish no keyed attribute when there is no key; remove
      `CUID_DEFAULT_SEED` and the default-restore branch.
- [x] 2.3 `cuid_seed` store persists to the variable (flags = provisioned)
      before installing the key; `-EIO` and no change on failure.
- [x] 2.4 `cuid_seed_state`: `unprovisioned` / `provisioned`.
- [ ] 2.5 Use `AMD_CUID_EFI_GUID` from `include/linux/efi.h`.
- [x] 2.6 ABI document; a KUnit known-answer test in amdgpu over a subset of
      the library's vectors.

## 3. Library, amd-smi, packaging

- [x] 3.1 Key source: `cuid_seed`, else the efivarfs payload (validated), else
      none; temporary CUIDs otherwise and for non-root callers.
- [x] 3.2 Key setting in `amdcuid_set_hash_key()`, used by amd-smi; exactly 32
      octets; refuse all-equal and public constants; sysfs first, efivarfs
      without amdgpu, a new variable created 0600. No standalone tool.
- [x] 3.3 Make the variable 0600 after setting the key only where efivarfs
      lists it with a wider mode, clearing the immutable flag first.
- [x] 3.4 tmpfiles.d entry for the variable, for kernels without the efivarfs
      table.
- [x] 3.5 `cuid_derived` first for GPUs and partitions.
- [x] 3.6 amd-smi seed-state strings, `set --cuid-seed`, key state in
      `static --cuid`, `list` and `node --cuid`, tmpfiles.d install from the
      amd-smi build.
- [ ] 3.7 An automated OVMF test for a driver-created variable and for the
      efivarfs path of `amdcuid_set_hash_key()`. Both are covered on hardware
      only.

## 4. Verification

- [x] 4.1 Radeon PRO W6800: the variable is created on first load, reused on
      reload, a `cuid_seed` write persists, and removal yields a fresh key.
- [x] 4.2 MI350X: SPX/DPX/QPX/CPX partition CUIDs pairwise distinct.
- [x] 4.3 efivarfs mode after remount on a kernel without the table (OVMF).
- [ ] 4.4 efivarfs mode with the table (OVMF): `AmdCuidKey` 0600 at mount,
      after a resume and when created through efivarfs.
- [x] 4.5 Radeon PRO W6800 with `cuid_derived`: library suite, amd-smi
      driver-published tests, with re-keying.
