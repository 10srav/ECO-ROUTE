"""
Unit tests for Energy Model module.
"""

import pytest
from controller.energy_model import EnergyModel, PortState


class TestEnergyModel:
    """Tests for EnergyModel class."""

    @pytest.fixture
    def energy_model(self):
        """Create an energy model instance for testing."""
        return EnergyModel(
            switch_base_power=50.0,
            port_power=5.0,
            sleep_power=0.5,
            wake_latency_ms=100.0
        )

    def test_initialization(self, energy_model):
        """Test energy model initialization."""
        assert energy_model.switch_base_power == 50.0
        assert energy_model.port_power == 5.0
        assert energy_model.sleep_power == 0.5
        assert energy_model.wake_latency_ms == 100.0

    def test_register_switch(self, energy_model):
        """Test switch registration."""
        ports = [1, 2, 3, 4]
        switch = energy_model.register_switch(1, ports)

        assert switch.dpid == 1
        assert len(switch.ports) == 4
        assert switch.base_power == 50.0
        # Total = 50 base + 4 ports * 5W = 70W
        assert switch.total_power == 70.0

    def test_set_port_sleeping(self, energy_model):
        """Test setting port to sleep."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        result = energy_model.set_port_sleeping(1, 1)

        assert result is True
        assert energy_model.is_port_sleeping(1, 1)
        assert not energy_model.is_port_active(1, 1)

    def test_set_port_active(self, energy_model):
        """Test waking up a sleeping port."""
        energy_model.register_switch(1, [1, 2, 3, 4])
        energy_model.set_port_sleeping(1, 1)

        result = energy_model.set_port_active(1, 1)

        assert result is True
        assert energy_model.is_port_active(1, 1)
        assert not energy_model.is_port_sleeping(1, 1)

    def test_power_calculation(self, energy_model):
        """Test power calculation with sleeping ports."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        # Initial: all active
        # Total = 50 + 4*5 = 70W
        snapshot1 = energy_model.calculate_snapshot()
        assert snapshot1.total_power == 70.0

        # Put 2 ports to sleep
        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_sleeping(1, 2)

        # Total = 50 + 2*5 + 2*0.5 = 61W
        snapshot2 = energy_model.calculate_snapshot()
        assert snapshot2.total_power == 61.0

    def test_energy_savings_calculation(self, energy_model):
        """Test energy savings percentage calculation."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        # Put 2 ports to sleep
        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_sleeping(1, 2)

        snapshot = energy_model.calculate_snapshot()

        # Baseline = 70W, Current = 61W
        # Savings = (70-61)/70 * 100 = 12.86%
        assert snapshot.energy_savings_percent > 0
        assert snapshot.energy_savings_percent < 100

    def test_link_energy_cost(self, energy_model):
        """Test link energy cost calculation."""
        energy_model.register_switch(1, [1, 2])
        energy_model.register_switch(2, [1, 2])

        # Active link - low cost
        cost_active = energy_model.get_link_energy_cost(1, 1, 2, 1)
        assert cost_active == 0.0

        # Sleep one port - higher cost
        energy_model.set_port_sleeping(1, 1)
        cost_sleeping = energy_model.get_link_energy_cost(1, 1, 2, 1)
        assert cost_sleeping > cost_active

    def test_path_energy_cost(self, energy_model):
        """Test path energy cost calculation."""
        energy_model.register_switch(1, [1, 2])
        energy_model.register_switch(2, [1, 2])
        energy_model.register_switch(3, [1, 2])

        path = [
            (1, 1, 2, 1),
            (2, 2, 3, 1)
        ]

        cost = energy_model.get_path_energy_cost(path)
        assert cost >= 0

    def test_active_ports_ratio(self, energy_model):
        """Test active ports ratio calculation."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        # All active
        ratio1 = energy_model.get_active_ports_ratio()
        assert ratio1 == 1.0

        # Put 2 to sleep
        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_sleeping(1, 2)

        ratio2 = energy_model.get_active_ports_ratio()
        assert ratio2 == 0.5

    def test_get_sleeping_links(self, energy_model):
        """Test getting list of sleeping links."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_sleeping(1, 3)

        sleeping = energy_model.get_sleeping_links()

        assert (1, 1) in sleeping
        assert (1, 3) in sleeping
        assert (1, 2) not in sleeping

    def test_get_active_links(self, energy_model):
        """Test getting list of active links."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        energy_model.set_port_sleeping(1, 1)

        active = energy_model.get_active_links()

        assert (1, 2) in active
        assert (1, 3) in active
        assert (1, 4) in active
        assert (1, 1) not in active

    def test_events_logging(self, energy_model):
        """Test sleep/wake event logging."""
        energy_model.register_switch(1, [1, 2])

        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_active(1, 1)

        events = energy_model.get_events()

        assert len(events) == 2
        assert events[0]["type"] == "port_sleep"
        assert events[1]["type"] == "port_wake"

    def test_snapshots_history(self, energy_model):
        """Test energy snapshots history."""
        energy_model.register_switch(1, [1, 2, 3, 4])

        for _ in range(5):
            energy_model.calculate_snapshot()

        snapshots = energy_model.get_snapshots()

        assert len(snapshots) == 5

    def test_switch_stats(self, energy_model):
        """Test per-switch statistics."""
        energy_model.register_switch(1, [1, 2, 3, 4])
        energy_model.set_port_sleeping(1, 1)

        stats = energy_model.get_switch_stats(1)

        assert stats is not None
        assert stats["dpid"] == 1
        assert stats["total_ports"] == 4
        assert stats["active_ports"] == 3
        assert stats["sleeping_ports"] == 1

    def test_unregister_switch(self, energy_model):
        """Test switch unregistration."""
        energy_model.register_switch(1, [1, 2])

        energy_model.unregister_switch(1)

        stats = energy_model.get_switch_stats(1)
        assert stats is None

    def test_reset(self, energy_model):
        """Test energy model reset."""
        energy_model.register_switch(1, [1, 2, 3, 4])
        energy_model.set_port_sleeping(1, 1)
        energy_model.calculate_snapshot()

        energy_model.reset()

        stats = energy_model.get_stats()
        assert stats["total_switches"] == 0

    def test_get_stats(self, energy_model):
        """Test comprehensive stats retrieval."""
        energy_model.register_switch(1, [1, 2, 3, 4])
        energy_model.register_switch(2, [1, 2, 3, 4])
        energy_model.set_port_sleeping(1, 1)
        energy_model.set_port_sleeping(2, 1)

        stats = energy_model.get_stats()

        assert stats["total_switches"] == 2
        assert stats["total_ports"] == 8
        assert stats["sleeping_ports"] == 2
        assert stats["active_ports"] == 6
        assert "energy_savings_percent" in stats
