// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Deref;
use std::os::unix::io::AsRawFd;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use super::RateLimiterConfig;
use crate::VmmError;
use crate::devices::virtio::device::VirtioDevice;
use crate::devices::virtio::net::{Net, TapError};
use crate::logger::warn;
use crate::rate_limiter::RateLimiter;
use crate::utils::net::mac::MacAddr;
use crate::vmm_config::enclave::DmbSizeMib;

/// This struct represents the strongly typed equivalent of the json body from net iface
/// related requests.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkInterfaceConfig {
    /// ID of the guest network interface.
    pub iface_id: String,
    /// Host level path for the guest network interface.
    pub host_dev_name: String,
    /// An open descriptor of a queue of `host_dev_name` that this process inherited. The
    /// device is used through it instead of being opened by name: nothing is attached to
    /// the device, and it keeps its queue when this process exits, for as long as the
    /// process that attached the queue holds its own copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_dev_fd: Option<i32>,
    /// Guest MAC address.
    pub guest_mac: Option<MacAddr>,
    /// Maximum Transmission Unit to advertise to the guest via VIRTIO_NET_F_MTU.
    pub mtu: Option<u16>,
    /// DMB size for confidential enclaves. Defaults to 16 MiB at attachment;
    /// transports without DMBs reject explicit sizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dmb_size_mib: Option<DmbSizeMib>,
    /// Rate Limiter for received packages.
    pub rx_rate_limiter: Option<RateLimiterConfig>,
    /// Rate Limiter for transmitted packages.
    pub tx_rate_limiter: Option<RateLimiterConfig>,
}

impl From<&Net> for NetworkInterfaceConfig {
    fn from(net: &Net) -> Self {
        let rx_rl: RateLimiterConfig = net.rx_rate_limiter().into();
        let tx_rl: RateLimiterConfig = net.tx_rate_limiter().into();
        NetworkInterfaceConfig {
            iface_id: net.id().to_string(),
            host_dev_name: net.iface_name(),
            // A descriptor names a queue in the process that holds it and nowhere else,
            // so a device's configuration does not carry one.
            host_dev_fd: None,
            guest_mac: net.guest_mac().copied(),
            mtu: net.mtu(),
            dmb_size_mib: net.dmb_size_mib(),
            rx_rate_limiter: rx_rl.into_option(),
            tx_rate_limiter: tx_rl.into_option(),
        }
    }
}

/// The data fed into a network iface update request. Currently, only the RX and TX rate limiters
/// can be updated.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkInterfaceUpdateConfig {
    /// The net iface ID, as provided by the user at iface creation time.
    pub iface_id: String,
    /// New RX rate limiter config. Only provided data will be updated. I.e. if any optional data
    /// is missing, it will not be nullified, but left unchanged.
    pub rx_rate_limiter: Option<RateLimiterConfig>,
    /// New TX rate limiter config. Only provided data will be updated. I.e. if any optional data
    /// is missing, it will not be nullified, but left unchanged.
    pub tx_rate_limiter: Option<RateLimiterConfig>,
}

/// Errors associated with the operations allowed on a net device.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum NetworkInterfaceError {
    /// Could not create the network device: {0}
    CreateNetworkDevice(#[from] crate::devices::virtio::net::NetError),
    /// Unable to update the net device: {0}
    DeviceUpdate(#[from] VmmError),
    /// The MAC address is already in use: {0}
    GuestMacAddressInUse(String),
    /// Descriptor {0} is already the TAP queue of network device '{1}'
    HostDevFdInUse(i32, String),
    /// Cannot open/create the tap device: {0}
    OpenTap(#[from] TapError),
}

/// Builder for a list of network devices.
#[derive(Debug, Default)]
pub struct NetBuilder {
    net_devices: Vec<Arc<Mutex<Net>>>,
}

impl NetBuilder {
    /// Creates an empty list of Network Devices.
    pub fn new() -> Self {
        NetBuilder {
            // List of built network devices.
            net_devices: Vec::new(),
        }
    }

    /// Returns a immutable iterator over the network devices.
    pub fn iter(&self) -> ::std::slice::Iter<'_, Arc<Mutex<Net>>> {
        self.net_devices.iter()
    }

    /// Adds an existing network device in the builder.
    pub fn add_device(&mut self, device: Arc<Mutex<Net>>) {
        self.net_devices.push(device);
    }

    /// Builds a network device based on a network interface config. Keeps a device reference
    /// in the builder's internal list.
    pub fn build(
        &mut self,
        netif_config: NetworkInterfaceConfig,
    ) -> Result<Arc<Mutex<Net>>, NetworkInterfaceError> {
        if let Some(ref mac_address) = netif_config.guest_mac {
            let mac_conflict = |net: &Arc<Mutex<Net>>| {
                let net = net.lock().expect("Poisoned lock");
                // Check if another net dev has same MAC.
                Some(mac_address) == net.guest_mac() && netif_config.iface_id != net.id()
            };
            // Validate there is no Mac conflict.
            // No need to validate host_dev_name conflict. In such a case,
            // an error will be thrown during device creation anyway.
            if self.net_devices.iter().any(mac_conflict) {
                return Err(NetworkInterfaceError::GuestMacAddressInUse(
                    mac_address.to_string(),
                ));
            }
        }

        // An inherited queue is one descriptor, and the device that takes it owns it: a
        // second device on the same number would read the same queue and close, when it is
        // dropped, whatever the number names by then. That includes the device this config
        // replaces, which closes the descriptor when it is removed below.
        if let Some(tap_fd) = netif_config.host_dev_fd {
            let holder = self.net_devices.iter().find_map(|net| {
                let net = net.lock().expect("Poisoned lock");
                (net.tap.as_raw_fd() == tap_fd).then(|| net.id().to_string())
            });
            if let Some(holder) = holder {
                return Err(NetworkInterfaceError::HostDevFdInUse(tap_fd, holder));
            }
        }

        // If this is an update, just remove the old one.
        if let Some(index) = self
            .net_devices
            .iter()
            .position(|net| net.lock().expect("Poisoned lock").id() == netif_config.iface_id)
        {
            warn!(
                "Replacing existing network device '{}'. The previous device configuration will \
                 be destroyed.",
                netif_config.iface_id
            );
            self.net_devices.swap_remove(index);
        }

        // Add new device.
        let net = Arc::new(Mutex::new(Self::create_net(netif_config)?));
        self.net_devices.push(net.clone());

        Ok(net)
    }

    /// Creates a Net device from a NetworkInterfaceConfig.
    pub fn create_net(cfg: NetworkInterfaceConfig) -> Result<Net, NetworkInterfaceError> {
        let rx_rate_limiter = cfg
            .rx_rate_limiter
            .map(RateLimiter::from)
            .unwrap_or_default();
        let tx_rate_limiter = cfg
            .tx_rate_limiter
            .map(RateLimiter::from)
            .unwrap_or_default();

        let net = match cfg.host_dev_fd {
            None => crate::devices::virtio::net::Net::new(
                cfg.iface_id,
                &cfg.host_dev_name,
                cfg.guest_mac,
                rx_rate_limiter,
                tx_rate_limiter,
                cfg.mtu,
            ),
            Some(tap_fd) => crate::devices::virtio::net::Net::new_inherited(
                cfg.iface_id,
                tap_fd,
                &cfg.host_dev_name,
                cfg.guest_mac,
                rx_rate_limiter,
                tx_rate_limiter,
                cfg.mtu,
            ),
        };
        let mut net = net.map_err(NetworkInterfaceError::CreateNetworkDevice)?;
        net.dmb_size_mib = cfg.dmb_size_mib;
        Ok(net)
    }

    /// Returns a vec with the structures used to configure the net devices.
    pub fn configs(&self) -> Vec<NetworkInterfaceConfig> {
        let mut ret = vec![];
        for net in &self.net_devices {
            ret.push(NetworkInterfaceConfig::from(net.lock().unwrap().deref()));
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use crate::rate_limiter::RateLimiter;

    impl NetBuilder {
        pub(crate) fn len(&self) -> usize {
            self.net_devices.len()
        }
    }

    fn create_netif(id: &str, name: &str, mac: &str) -> NetworkInterfaceConfig {
        NetworkInterfaceConfig {
            iface_id: String::from(id),
            host_dev_name: String::from(name),
            host_dev_fd: None,
            guest_mac: Some(MacAddr::from_str(mac).unwrap()),
            mtu: None,
            dmb_size_mib: None,
            rx_rate_limiter: RateLimiterConfig::default().into_option(),
            tx_rate_limiter: RateLimiterConfig::default().into_option(),
        }
    }

    /// A config that names an inherited queue builds its device on that queue: the TAP
    /// device stays attached once, by its holder, and a queue of another device is refused.
    #[test]
    fn test_create_net_on_an_inherited_queue() {
        use crate::devices::virtio::net::Tap;

        let holder = Tap::open_named("inheritnet").unwrap();
        let mut cfg = create_netif("id", "inheritnet", "01:23:45:67:89:0a");
        // SAFETY: dup of a descriptor this test owns.
        cfg.host_dev_fd = Some(unsafe { libc::dup(holder.as_raw_fd()) });
        let net = NetBuilder::create_net(cfg).unwrap();
        assert_eq!(net.iface_name(), "inheritnet");
        Tap::open_named("inheritnet").unwrap_err();
        drop(net);

        let mut wrong = create_netif("id", "inheritother", "01:23:45:67:89:0a");
        // SAFETY: as above; closed below, since a refused descriptor stays open.
        let fd = unsafe { libc::dup(holder.as_raw_fd()) };
        wrong.host_dev_fd = Some(fd);
        NetBuilder::create_net(wrong).unwrap_err();
        // SAFETY: the descriptor is this test's own.
        unsafe { libc::close(fd) };
    }

    /// One inherited queue is one device's: a second config that names the same descriptor
    /// is refused, whether it is for another device or replaces the one that has it, and
    /// the descriptor stays the first device's.
    #[test]
    fn test_an_inherited_queue_is_one_devices() {
        use crate::devices::virtio::net::Tap;

        let holder = Tap::open_named("inheritonce").unwrap();
        // SAFETY: dup of a descriptor this test owns; the device built on it owns the dup.
        let fd = unsafe { libc::dup(holder.as_raw_fd()) };
        let mut builder = NetBuilder::new();
        let mut first = create_netif("first", "inheritonce", "01:23:45:67:89:0a");
        first.host_dev_fd = Some(fd);
        let net = builder.build(first).unwrap();
        assert_eq!(net.lock().unwrap().tap.as_raw_fd(), fd);

        for (id, mac) in [
            ("second", "01:23:45:67:89:0b"),
            ("first", "01:23:45:67:89:0a"),
        ] {
            let mut again = create_netif(id, "inheritonce", mac);
            again.host_dev_fd = Some(fd);
            match builder.build(again) {
                Err(NetworkInterfaceError::HostDevFdInUse(refused, by)) => {
                    assert_eq!((refused, by.as_str()), (fd, "first"), "{id}");
                }
                other => panic!("{id}: {other:?}"),
            }
        }
        assert_eq!(builder.len(), 1);
        // The first device still has its queue: the descriptor is open and attached.
        // SAFETY: F_GETFD takes no pointer.
        assert!(unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0);
    }

    impl Clone for NetworkInterfaceConfig {
        fn clone(&self) -> Self {
            NetworkInterfaceConfig {
                iface_id: self.iface_id.clone(),
                host_dev_name: self.host_dev_name.clone(),
                host_dev_fd: self.host_dev_fd,
                guest_mac: self.guest_mac,
                mtu: self.mtu,
                dmb_size_mib: self.dmb_size_mib,
                rx_rate_limiter: None,
                tx_rate_limiter: None,
            }
        }
    }

    #[test]
    fn test_insert() {
        let mut net_builder = NetBuilder::new();

        let id_1 = "id_1";
        let mut host_dev_name_1 = "dev1";
        let mut guest_mac_1 = "01:23:45:67:89:0a";

        // Test create.
        let mut netif_1 = create_netif(id_1, host_dev_name_1, guest_mac_1);
        netif_1.dmb_size_mib = Some(DmbSizeMib::try_from(64).unwrap());
        net_builder.build(netif_1).unwrap();
        assert_eq!(net_builder.net_devices.len(), 1);
        assert_eq!(net_builder.configs()[0].dmb_size_mib.unwrap().get(), 64);

        // Test update mac address (this test does not modify the tap).
        guest_mac_1 = "01:23:45:67:89:0b";
        let netif_1 = create_netif(id_1, host_dev_name_1, guest_mac_1);

        net_builder.build(netif_1).unwrap();
        assert_eq!(net_builder.net_devices.len(), 1);
        assert_eq!(net_builder.configs()[0].dmb_size_mib, None);

        // Test update host_dev_name (the tap will be updated).
        host_dev_name_1 = "dev2";
        let netif_1 = create_netif(id_1, host_dev_name_1, guest_mac_1);
        net_builder.build(netif_1).unwrap();
        assert_eq!(net_builder.net_devices.len(), 1);
    }

    #[test]
    fn test_insert_error_cases() {
        let mut net_builder = NetBuilder::new();

        let id_1 = "id_1";
        let host_dev_name_1 = "dev3";
        let guest_mac_1 = "01:23:45:67:89:0a";

        // Adding the first valid network config.
        let netif_1 = create_netif(id_1, host_dev_name_1, guest_mac_1);
        net_builder.build(netif_1).unwrap();

        // Error Cases for CREATE
        // Error Case: Add new network config with the same mac as netif_1.
        let id_2 = "id_2";
        let host_dev_name_2 = "dev4";
        let guest_mac_2 = "01:23:45:67:89:0b";

        let netif_2 = create_netif(id_2, host_dev_name_2, guest_mac_1);
        let expected_error = NetworkInterfaceError::GuestMacAddressInUse(guest_mac_1.into());
        assert_eq!(
            net_builder.build(netif_2).err().unwrap().to_string(),
            expected_error.to_string()
        );
        assert_eq!(net_builder.net_devices.len(), 1);

        // Error Case: Add new network config with the same dev_host_name as netif_1.
        let netif_2 = create_netif(id_2, host_dev_name_1, guest_mac_2);
        assert_eq!(
            net_builder.build(netif_2).err().unwrap().to_string(),
            NetworkInterfaceError::CreateNetworkDevice(
                crate::devices::virtio::net::NetError::TapOpen(TapError::IfreqExecuteError(
                    std::io::Error::from_raw_os_error(16),
                    host_dev_name_1.to_string()
                ))
            )
            .to_string()
        );
        assert_eq!(net_builder.net_devices.len(), 1);

        // Adding the second valid network config.
        let netif_2 = create_netif(id_2, host_dev_name_2, guest_mac_2);
        net_builder.build(netif_2).unwrap();

        // Error Cases for UPDATE
        // Error Case: Update netif_2 mac using the same mac as netif_1.
        let netif_2 = create_netif(id_2, host_dev_name_2, guest_mac_1);
        let expected_error = NetworkInterfaceError::GuestMacAddressInUse(guest_mac_1.into());
        assert_eq!(
            net_builder.build(netif_2).err().unwrap().to_string(),
            expected_error.to_string()
        );

        // Error Case: Update netif_2 dev_host_name using the same dev_host_name as netif_1.
        let netif_2 = create_netif(id_2, host_dev_name_1, guest_mac_2);
        assert_eq!(
            net_builder.build(netif_2).err().unwrap().to_string(),
            NetworkInterfaceError::CreateNetworkDevice(
                crate::devices::virtio::net::NetError::TapOpen(TapError::IfreqExecuteError(
                    std::io::Error::from_raw_os_error(16),
                    host_dev_name_1.to_string()
                ))
            )
            .to_string()
        );
    }

    #[test]
    fn test_net_config() {
        let net_id = "id";
        let host_dev_name = "dev";
        let guest_mac = "01:23:45:67:89:0b";

        let net_if_cfg = create_netif(net_id, host_dev_name, guest_mac);
        assert_eq!(
            net_if_cfg.guest_mac.unwrap(),
            MacAddr::from_str(guest_mac).unwrap()
        );

        let mut net_builder = NetBuilder::new();
        net_builder.build(net_if_cfg.clone()).unwrap();
        assert_eq!(net_builder.net_devices.len(), 1);

        let configs = net_builder.configs();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs.first().unwrap(), &net_if_cfg);
    }

    #[test]
    fn test_add_device() {
        let mut net_builder = NetBuilder::new();
        let net_id = "test_id";
        let host_dev_name = "dev";
        let guest_mac = "01:23:45:67:89:0b";

        let net = Net::new(
            net_id.to_string(),
            host_dev_name,
            Some(MacAddr::from_str(guest_mac).unwrap()),
            RateLimiter::default(),
            RateLimiter::default(),
            None,
        )
        .unwrap();

        net_builder.add_device(Arc::new(Mutex::new(net)));
        assert_eq!(net_builder.net_devices.len(), 1);
        assert_eq!(net_builder.configs()[0].dmb_size_mib, None);
        assert_eq!(
            net_builder
                .net_devices
                .pop()
                .unwrap()
                .lock()
                .unwrap()
                .deref()
                .id(),
            net_id
        );
    }
}
