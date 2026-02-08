#!/usr/bin/env python3
"""
Launcher for EcoRoute SDN Controller.

Replaces osken-manager / ryu-manager CLI for environments where
the CLI entry point is not installed (e.g. os-ken on Python 3.12).

Usage:
    python start_controller.py
"""
import gc
import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller.compat import app_manager, hub, wsgi, _FRAMEWORK

# Import cfg based on framework
if _FRAMEWORK == "os_ken":
    from os_ken import cfg
else:
    from ryu import cfg

AppManager = app_manager.AppManager
start_service = wsgi.start_service

# Configure logging BEFORE cfg.CONF so warning handlers are available
logging.basicConfig(level=logging.INFO)

# Initialize os_ken CONF with defaults before any app loads.
# Required for WSGI server (wsapi-host / wsapi-port) and other config opts.
try:
    cfg.CONF(args=[], project='os_ken')
except SystemExit as e:
    logging.getLogger(__name__).warning("CONF init raised SystemExit (code=%s), continuing", e.code)
    if e.code not in (None, 0):
        raise


def main():
    app_lists = [
        'controller.ecoroute_controller',
        'os_ken.topology.switches',
    ]

    app_mgr = AppManager.get_instance()
    app_mgr.load_apps(app_lists)
    contexts = app_mgr.create_contexts()
    services = app_mgr.instantiate_apps(**contexts)

    # Start WSGI REST API server (port 8080) — not auto-started by native hub
    wsgi_server = start_service(app_mgr)
    if wsgi_server:
        wsgi_thread = hub.spawn(wsgi_server)
        services.append(wsgi_thread)

    try:
        hub.joinall(services)
    finally:
        # Stop services first, then close the app manager
        for t in services:
            if hasattr(t, 'kill'):
                t.kill()
            elif hasattr(t, 'cancel'):
                t.cancel()
        hub.joinall(services)
        app_mgr.close()
        gc.collect()


if __name__ == '__main__':
    main()
