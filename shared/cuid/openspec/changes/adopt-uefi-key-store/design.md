# Design: a UEFI-backed node key

## Context

The derived CUID of every component on a host is `HMAC-SHA256(key, primary)`.
Until now the key had two owners: amdgpu held an in-memory copy that fell back
to a public default seed, and the library kept a key file, a record store and a
daemon that reconciled the two.

`split-identity-from-key-store` removed the library's copy, the daemon and the
default seed, and specified everything that needs no key: `cuid_primary`,
`cuid_unit_id`, the XCC-range partition UnitID and temporary CUIDs. This change
gives the key one owner, the host's UEFI variable store, and with it the
derived CUIDs that are not temporary.

## Goals

- One authoritative key per host, readable by the driver before user space runs.
- No public default key, and no CUID derived under a key nobody chose or generated.
- The key is readable by root only, on every kernel that carries this change.
- An unprivileged caller, or a host without a key, still gets a usable
  temporary CUID that is marked as temporary.

## Non-goals

- An out-of-band implementation. The variable format is the contract it will use.
- Per-VM serials and a virtualized key store. These belong to the hypervisor.
- Non-AMD devices and an OS-level CUID service.

## Decisions

### The key store

One variable per host:

| Field | Value |
|---|---|
| Name | `AmdCuidKey` |
| Vendor GUID | `e41c1f7f-63cb-46b9-bf27-36a55f92a06d` |
| Attributes | `NON_VOLATILE \| BOOTSERVICE_ACCESS \| RUNTIME_ACCESS` (0x7), never authenticated |
| Payload | 36 octets: `version` (1 octet, = 1), `flags` (1 octet, bit 0 = provisioned by an administrator, other bits zero), 2 reserved octets (zero), 32-octet key |

A variable with other attributes, or a payload of any other size, version,
flags or reserved value, is malformed. Nobody
overwrites a malformed variable automatically: the driver publishes no derived
CUID and logs once.

**Driver flow**, once per module lifetime, at the first CUID registration and
outside `cuid_seed_lock`:

1. `!IS_ENABLED(CONFIG_EFI)` or `!efivar_is_available()` → no key.
2. `efivar_lock()`; `efivar_get_variable()`.
3. `EFI_NOT_FOUND` → `get_random_bytes(32)`, flags = 0,
   `efivar_set_variable_locked(..., nonblocking=false)`. Any write error → no key.
4. Any other read error, or a malformed payload → no key.
5. `efivar_unlock()`. The result, key or no key, is cached until module unload.

`efi=noruntime`, `CONFIG_EFI_DISABLE_RUNTIME`, hypervisors without a runtime
variable store and non-UEFI boots all stop at step 1.

**Without a key the driver publishes no keyed attribute**: no `cuid_derived`,
`cuid_seed` or `cuid_seed_state`, on the device or its partitions. What it
publishes without a key is the identity of `split-identity-from-key-store`.
Consumers have one rule: no `cuid_derived` means the library derives the CUID
itself where it holds the key, and a temporary CUID otherwise. The probe never
fails and never defers because of CUID.

**`cuid_seed` store** (`CAP_SYS_ADMIN`, exactly 32 octets): the driver writes
the variable first, with flags = provisioned, and only on success installs the
key and re-keys every component. A failed write returns `-EIO` and changes
nothing, because the firmware copy is authoritative. There is no way back to
`unprovisioned` other than deleting the variable while amdgpu is unloaded.

**`cuid_seed_state`** reports `unprovisioned` for a key the driver generated
and `provisioned` once flags bit 0 is set.

**Hosts without amdgpu.** Nothing creates the key implicitly: a lookup never
writes efivarfs. Without the variable such a host has temporary CUIDs only.

**Setting the key and listing CUIDs.** There is no standalone tool. amd-smi is
the administrator's interface: `amd-smi set --cuid-seed <file|->` sets the key,
and `amd-smi node --cuid` lists every component's CUID through
`amdsmi_get_cuid_components()`, including the CPUs, NICs and platform amd-smi
does not otherwise manage. A program calls `amdcuid_set_hash_key()`, which
amd-smi uses. The call:

- accepts exactly 32 octets;
- refuses a key whose octets are all equal, or that equals a published constant
  zero-padded (`AMD-CUID-DEFAULT-SEED-v1`, `AMD-CUID-TEMP-KEY-v1`, the test keys
  in `cuid_vectors.txt`); the kernel's `cuid_seed` does not refuse them, so
  the check is in user space only;
- writes `cuid_seed` on one device when amdgpu is loaded, so the driver
  persists it; otherwise writes the efivarfs variable directly (4-octet
  attribute header and the 36-octet payload, flags = provisioned), creating
  it with `open(O_CREAT, 0600)` when it does not exist;
- where efivarfs lists the variable with a wider mode, clears the immutable
  flag, makes it 0600 and restores the flag, warning rather than failing if it
  cannot.

Every derived CUID on the host changes. The library is a static archive,
`libamdcuid_static.a`, with one header and a CMake package, so any program can
link it without a runtime dependency; amd-smi absorbs it.

### Confidentiality of the variable

efivarfs gives every variable mode 0644, readable by every local user, and
recreates it 0644 at every mount. This change adds a table of variables whose
contents are secret to efivarfs, matched by {GUID, name} the way
`variable_validate[]` is, and efivarfs creates their inodes 0600 both when it
populates the filesystem and when a variable is created through it. The first
entry is systemd's `LoaderSystemToken`, which has the same bug; the second is
`AmdCuidKey`, whose GUID becomes `AMD_CUID_EFI_GUID` in `include/linux/efi.h`
and which amdgpu then uses. Root can still read and write the variable; see
`split-identity-from-key-store` for the alternatives.

Two things efivarfs does not change:

- A variable outside `variable_validate[]` is immutable. A variable created
  through efivarfs is created with the mode `open()` asks for, which the
  library sets to 0600 and the table forces to 0600, but a later `chmod()`
  fails with `EPERM` until the flag is cleared.
- efivarfs enumerates variables at mount, and again at a resume from
  hibernation. One the driver creates after that is not listed until then.

On a kernel without the efivarfs table, confidentiality comes from user space:

1. The package ships `tmpfiles.d/amdcuid.conf`:
   ```
   h /sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d - - - - -i
   z /sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d 0600 root root -
   ```
   The `h … -i` line clears the immutable flag, since a bare `chmod` fails with
   `EPERM`. The variable then stays mutable, so root could remove it, but root
   can already rewrite it. The rule takes effect from the boot after the
   variable first appears. On a kernel with the table the `z` line finds the
   variable 0600 already.
2. The library's key-setting call makes the variable 0600 right after it
   writes, as above, and warns rather than fails if it cannot.
3. The variable is 0644 until one of those runs: early in every boot, before
   `systemd-tmpfiles-setup`; after a resume from hibernation, until the next
   boot; on hosts without systemd-tmpfiles; and on hosts without the ROCm
   package, where the driver created a per-host key. A per-host random key read
   locally reveals nothing that `cuid_derived` (0444) does not; a fleet key read
   on one host is the real risk, so the key-setting call warns when it cannot
   make the variable 0600.

Rejected: keeping the fleet key somewhere else (two stores again), accepting
the exposure, and a boot-services-only variable handed over before boot.

### Library

- **Key, root only**: `cuid_seed` of any amdgpu device, which is the driver's
  key by construction; else the efivarfs variable with its header stripped and
  its payload validated; else none.
- **No key, or not root** → temporary CUID, as in
  `split-identity-from-key-store`. The same CPU therefore has a permanent CUID
  for root and a temporary one for other users; bit 117 and the reported source
  tell them apart.
- **GPUs and partitions**: the driver's `cuid_derived` first, with source
  `DRIVER`, for every caller. Without it, a whole GPU whose driver publishes
  `cuid_primary` and `cuid_unit_id` is keyed like a CPU; a partition is
  refused.
- **amd-smi handles**: on a GPU that publishes a partition node in every
  compute mode, each handle reports its partition's derived CUID. In SPX that
  is the SPX partition's (UnitID `0x200` on 8 XCCs), which differs from the
  whole GPU's (UnitID 0); `amd-smi node --cuid` lists the two separately. The
  whole GPU's CUID stands in for an SPX partition only where the partition has
  none, as `split-identity-from-key-store` specifies.
- A lookup reads the key once and uses it for every property query on the
  handles it returns; a key changed later takes effect at the next lookup or
  `amdcuid_refresh()`.
- amd-smi's seed-state strings follow the kernel.

### Virtualization

- Under full passthrough, the guest's amdgpu sees the physical serial and the
  guest's own variable store, so it creates its own key. Primary CUIDs match
  the host's; derived CUIDs differ unless the same key is set in the guest.
  The key belongs to a variable store, so this is intended.
- A guest without a runtime variable store publishes no keyed attribute and
  uses temporary CUIDs.

### Tests

`key_store_test` covers the 36-octet payload parse cases and the refused keys.

## Risks and trade-offs

- **Local exposure before the variable is 0600,** on a kernel without the
  efivarfs table. See "Confidentiality".
- **The efivarfs table may be asked to change shape** (for example a
  Linux-owned GUID); the amdgpu commit then follows it.
- **A malformed variable blocks every derived CUID on the host** that is not
  temporary, until an administrator deletes or rewrites it. Overwriting it automatically would silently change
  every derived CUID.
- **Re-keying changes every derived CUID on the host.** There is no mapping
  from old values to new ones.
- **Root and other users see different CPU, NIC and platform CUIDs.** Bit 117
  and the source say which one a caller got.

## Open questions

- Per-VM serials and a virtualized `AmdCuidKey` in hypervisors.
