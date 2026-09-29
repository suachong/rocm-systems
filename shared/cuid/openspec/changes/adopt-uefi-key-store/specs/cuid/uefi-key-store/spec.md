## ADDED Requirements

### Requirement: The host key lives in one UEFI variable

The node key for every derived CUID on a host SHALL be stored in the UEFI
variable `AmdCuidKey` with vendor GUID `e41c1f7f-63cb-46b9-bf27-36a55f92a06d`
and attributes NON_VOLATILE, BOOTSERVICE_ACCESS and RUNTIME_ACCESS. Its payload
SHALL be exactly 36 octets: a version octet equal to 1, a flags octet whose bit
0 means "provisioned by an administrator" and whose other bits are zero, two
zero octets, and the 32-octet key. Any other payload is malformed.

No other location SHALL hold the key persistently.

#### Scenario: Well-formed payload

- **WHEN** the variable holds 36 octets starting `01 00 00 00`
- **THEN** the last 32 octets are the key and the state is unprovisioned

#### Scenario: Malformed payload is left alone

- **WHEN** the variable holds 32 octets, or a version other than 1
- **THEN** no producer overwrites it automatically
- **AND** no permanent CUID is produced

### Requirement: The driver creates the key when it is absent

On the first CUID registration after the driver loads, the driver SHALL read
the variable and, when the firmware reports it absent, SHALL generate 32 random
octets and write them with flags zero, all under the variable-store lock so that
two writers cannot race. The result SHALL be cached until the driver unloads.

When runtime variable services are unavailable, or the read or write fails for
any other reason, the driver SHALL publish no `cuid_derived`, `cuid_seed` or
`cuid_seed_state` and SHALL NOT fail or defer the device probe.

#### Scenario: First load on a fresh host

- **WHEN** amdgpu loads on a UEFI host without `AmdCuidKey`
- **THEN** the variable exists afterwards with a random key
- **AND** `cuid_seed_state` reads `unprovisioned`

#### Scenario: Reload reuses the key

- **WHEN** amdgpu is unloaded and loaded again
- **THEN** every `cuid_derived` is unchanged

#### Scenario: No runtime services

- **WHEN** the host booted with `efi=noruntime`
- **THEN** no device or partition has a `cuid_derived`

### Requirement: Only the driver or an explicit key-setting call writes the variable

The user-space library SHALL NOT create or modify the variable except in
`amdcuid_set_hash_key()`, the call amd-smi's `set --cuid-seed` uses. That call
SHALL set the key through `cuid_seed` when an amdgpu `cuid_seed` attribute
exists, and MAY write the variable directly only when none does. A lookup
SHALL NOT write it.

#### Scenario: Setting the key without amdgpu

- **WHEN** root calls `amdcuid_set_hash_key()` with an acceptable key on a host
  with efivarfs and no amdgpu
- **THEN** the variable holds that key with flags bit 0 set
- **AND** amdgpu, loaded afterwards, reports `provisioned` and uses that key

#### Scenario: Host without amdgpu

- **WHEN** a root library lookup runs on a host with no variable and no amdgpu
- **THEN** the variable still does not exist afterwards
- **AND** CPU, NIC and platform identities are temporary CUIDs

### Requirement: The variable is not world-readable

efivarfs SHALL create the inode of a variable listed in its table of secret
variables with mode 0600, both when it populates the filesystem and when the
variable is created through it, and `AmdCuidKey` under its vendor GUID SHALL be
in that table. Other variables SHALL keep mode 0644.

`amdcuid_set_hash_key()` SHALL create the variable with mode 0600, and SHALL
make an existing one 0600 where efivarfs lists it with a wider mode, clearing
the immutable flag first. For kernels without the efivarfs table, packaging
SHALL install a tmpfiles.d entry that clears the immutable flag on the variable
and sets mode 0600 at boot.

#### Scenario: Mode after boot

- **WHEN** a host with the variable boots
- **THEN** `/sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d`
  is not readable by an unprivileged user

#### Scenario: Created through efivarfs

- **WHEN** root creates the variable through efivarfs, under any umask
- **THEN** its mode is 0600

#### Scenario: Other variables

- **WHEN** efivarfs lists `BootOrder`
- **THEN** its mode is 0644
