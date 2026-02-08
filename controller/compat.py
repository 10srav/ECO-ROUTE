"""
Compatibility shim for os-ken / ryu imports.

os-ken is the maintained fork of ryu. Both expose identical APIs under
different package names. This module detects which is installed and
re-exports the required submodules so the rest of the codebase can do:

    from controller.compat import app_manager, ofp_event, handler, hub, ...

without caring whether os-ken or ryu is actually installed.
"""

import importlib
import sys

# Detect which SDN framework is available
_FRAMEWORK = None

try:
    import os_ken  # noqa: F401
    _FRAMEWORK = "os_ken"
except ImportError:
    try:
        import ryu  # noqa: F401
        _FRAMEWORK = "ryu"
    except ImportError:
        raise ImportError(
            "Neither os-ken nor ryu is installed. "
            "Install one with: pip install os-ken  OR  pip install ryu"
        )


def _import(submodule: str):
    """Import a submodule from whichever framework is available."""
    return importlib.import_module(f"{_FRAMEWORK}.{submodule}")


# --- Core framework modules ---
app_manager = _import("base.app_manager")
ofp_event = _import("controller.ofp_event")
handler = _import("controller.handler")
hub = _import("lib.hub")
ofproto_v1_3 = _import("ofproto.ofproto_v1_3")

# handler constants (re-export for convenience)
CONFIG_DISPATCHER = handler.CONFIG_DISPATCHER
DEAD_DISPATCHER = handler.DEAD_DISPATCHER
MAIN_DISPATCHER = handler.MAIN_DISPATCHER
set_ev_cls = handler.set_ev_cls

# --- Packet library ---
packet_lib = _import("lib.packet.packet")
ethernet = _import("lib.packet.ethernet")
arp = _import("lib.packet.arp")
ipv4 = _import("lib.packet.ipv4")
icmp = _import("lib.packet.icmp")
tcp = _import("lib.packet.tcp")
udp = _import("lib.packet.udp")
lldp = _import("lib.packet.lldp")

# --- Topology ---
topo_event = _import("topology.event")
topo_api = _import("topology.api")
get_all_link = topo_api.get_all_link
get_all_switch = topo_api.get_all_switch

# --- WSGI ---
wsgi = _import("app.wsgi")
ControllerBase = wsgi.ControllerBase
WSGIApplication = wsgi.WSGIApplication
route = wsgi.route

# --- Optional: cfg for start_controller.py ---
try:
    cfg = _import("cfg") if _FRAMEWORK == "os_ken" else None
except Exception:
    cfg = None
