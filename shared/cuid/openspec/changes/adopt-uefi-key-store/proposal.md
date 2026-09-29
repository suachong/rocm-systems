## Why

A derived CUID that is not temporary needs a node key that other local users
cannot read. `split-identity-from-key-store` removed the daemon, the
`/etc/amdcuid` key file and the public default seed, and left every derived
CUID temporary. This change makes the host's UEFI variable store the single
authoritative key store, owned by the driver.

Every implemented layer contradicted that: the kernel fell back to
`AMD-CUID-DEFAULT-SEED-v1`, the library owned the key file, the daemon
reconciled driver seeds, a TPM-sealed variable was offered that the driver
cannot read, and efivarfs showed any UEFI variable to every local user.

## What Changes

**efivarfs**

- efivarfs creates variables that hold secrets mode 0600, both when it
  populates the filesystem and when a variable is created through it, from a
  table of {GUID, name} beside `variable_validate[]`. The first entry is
  systemd's `LoaderSystemToken`; the second is `AmdCuidKey`, whose GUID is
  `AMD_CUID_EFI_GUID` in `include/linux/efi.h`.

**Kernel (`amdgpu`)**

- **BREAKING** The node key comes from the UEFI variable `AmdCuidKey`
  (GUID `e41c1f7f-63cb-46b9-bf27-36a55f92a06d`). The driver reads it at the first
  CUID registration and, if absent, generates 32 random octets and writes it.
- **BREAKING** Without a readable or creatable variable the driver publishes no
  keyed attribute. `CUID_DEFAULT_SEED` and the padded-default restore are
  removed.
- `cuid_derived` (0444), `cuid_seed` (0600) and `cuid_seed_state` (0444) beside
  the identity attributes.
- A `cuid_seed` write persists the key to the variable before re-keying; a
  failed write changes nothing.
- **BREAKING** `cuid_seed_state` reports `unprovisioned` or `provisioned`.

**Library and packaging (`shared/cuid`)**

- Root obtains the key from `cuid_seed`, else the efivarfs variable; without a
  key, and for every non-root caller, identities stay temporary CUIDs.
- The library reads `cuid_derived` first for GPUs and partitions.
- amd-smi (`set --cuid-seed`) and `amdcuid_set_hash_key()` set the key, through
  `cuid_seed` or, without amdgpu, efivarfs; a key must be exactly 32 octets,
  and trivial and public keys are refused. A variable the call creates is
  created mode 0600.
- Packaging ships a tmpfiles.d entry that makes the variable 0600 at boot on a
  kernel without the efivarfs table.

**amd-smi**: seed-state strings, `set --cuid-seed`, and the key state in
`static --cuid`, `list` and `node --cuid`.

## Impact

- Builds on `split-identity-from-key-store`, which specifies the identity
  attributes, the driver-published marker `cuid_unit_id`, the partition UnitID,
  temporary CUIDs, AMD-only GPU listing and the removal of the daemon.
- Supersedes, in other open changes: `sysfs-interface` "Default seed" and the
  `default|custom` states; `library-consumer` key handling;
  `amdsmi-seed-provisioning` "A drifted driver seed is repaired".
- Restores, after `split-identity-from-key-store` withdrew them,
  `amdsmi-seed-provisioning` and the `set --cuid-seed` requirement of
  `amdsmi-cli` in `integrate-cuid-into-amdsmi`.
- Code: `fs/efivarfs`, `include/linux/efi.h`, kernel `amdgpu_cuid.{c,h}`, ABI
  doc, KUnit HMAC vectors; `shared/cuid/{lib,scripts,tests}`; `projects/amdsmi`.
- Design and every decision: `design.md`.
