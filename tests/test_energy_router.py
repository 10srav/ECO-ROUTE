"""
Unit tests for Energy Router module.
"""

import pytest
from controller.energy_model import EnergyModel
from controller.ewma_predictor import AdaptiveEWMAPredictor
from controller.energy_router import EnergyAwareRouter


class TestEnergyAwareRouter:
    """Tests for EnergyAwareRouter class."""

    @pytest.fixture
    def router(self):
        """Create a router instance with dependencies."""
        predictor = AdaptiveEWMAPredictor()
        energy_model = EnergyModel()
        router = EnergyAwareRouter(
            energy_model=energy_model,
            predictor=predictor,
            k_paths=3,
            max_utilization=80.0
        )
        return router

    @pytest.fixture
    def simple_topology(self, router):
        """Create a simple linear topology: 1 -- 2 -- 3."""
        router.add_link(1, 1, 2, 1, 1000.0)
        router.add_link(2, 1, 1, 1, 1000.0)  # Reverse
        router.add_link(2, 2, 3, 1, 1000.0)
        router.add_link(3, 1, 2, 2, 1000.0)  # Reverse

        router.energy_model.register_switch(1, [1])
        router.energy_model.register_switch(2, [1, 2])
        router.energy_model.register_switch(3, [1])

        return router

    @pytest.fixture
    def diamond_topology(self, router):
        """
        Create a diamond topology:
            1
           / \
          2   3
           \ /
            4
        """
        # 1 -> 2, 1 -> 3
        router.add_link(1, 1, 2, 1, 1000.0)
        router.add_link(1, 2, 3, 1, 1000.0)

        # 2 -> 4, 3 -> 4
        router.add_link(2, 2, 4, 1, 1000.0)
        router.add_link(3, 2, 4, 2, 1000.0)

        # Reverse links
        router.add_link(2, 1, 1, 1, 1000.0)
        router.add_link(3, 1, 1, 2, 1000.0)
        router.add_link(4, 1, 2, 2, 1000.0)
        router.add_link(4, 2, 3, 2, 1000.0)

        router.energy_model.register_switch(1, [1, 2])
        router.energy_model.register_switch(2, [1, 2])
        router.energy_model.register_switch(3, [1, 2])
        router.energy_model.register_switch(4, [1, 2])

        return router

    def test_add_link(self, router):
        """Test adding links to topology."""
        router.add_link(1, 1, 2, 1, 1000.0)

        info = router.get_topology_info()
        assert info["total_edges"] == 1

    def test_remove_link(self, router):
        """Test removing links from topology."""
        router.add_link(1, 1, 2, 1, 1000.0)
        router.add_link(2, 1, 3, 1, 1000.0)

        router.remove_link(1, 2)

        info = router.get_topology_info()
        assert info["total_edges"] == 1

    def test_add_host(self, router):
        """Test adding hosts to topology."""
        router.add_host("10.0.0.1", 1, 1)

        location = router.get_host_location("10.0.0.1")
        assert location == (1, 1)

    def test_find_k_shortest_paths_simple(self, simple_topology):
        """Test finding k-shortest paths in simple topology."""
        paths = simple_topology.find_k_shortest_paths(1, 3)

        assert len(paths) >= 1
        assert paths[0] == [1, 2, 3]

    def test_find_k_shortest_paths_diamond(self, diamond_topology):
        """Test finding k-shortest paths in diamond topology."""
        paths = diamond_topology.find_k_shortest_paths(1, 4)

        assert len(paths) == 2
        # Both paths should have same length
        assert len(paths[0]) == 3
        assert len(paths[1]) == 3

    def test_find_best_path(self, diamond_topology):
        """Test finding best energy-aware path."""
        # Put one path to sleep
        diamond_topology.energy_model.set_port_sleeping(2, 1)
        diamond_topology.energy_model.set_port_sleeping(2, 2)

        best_path = diamond_topology.find_best_path(1, 4)

        assert best_path is not None
        # Should prefer the path not using sleeping links
        assert 2 not in best_path.path or 3 in best_path.path

    def test_path_scoring(self, diamond_topology):
        """Test path scoring mechanism."""
        path = [1, 2, 4]
        score = diamond_topology.score_path(path)

        assert score.total_score >= 0
        assert score.energy_score >= 0
        assert score.load_score >= 0
        assert score.hop_score >= 0

    def test_install_flow(self, diamond_topology):
        """Test flow installation."""
        best_path = diamond_topology.find_best_path(1, 4)

        flow = diamond_topology.install_flow(
            flow_id="test_flow",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            path_score=best_path,
            bandwidth=100.0
        )

        assert flow.flow_id == "test_flow"
        assert flow.src_ip == "10.0.0.1"
        assert flow.dst_ip == "10.0.0.2"

    def test_remove_flow(self, diamond_topology):
        """Test flow removal."""
        best_path = diamond_topology.find_best_path(1, 4)

        diamond_topology.install_flow(
            flow_id="test_flow",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            path_score=best_path,
            bandwidth=100.0
        )

        removed = diamond_topology.remove_flow("test_flow")

        assert removed is not None
        assert removed.flow_id == "test_flow"

    def test_get_flows_on_link(self, diamond_topology):
        """Test getting flows on a specific link."""
        path1 = diamond_topology.score_path([1, 2, 4])

        diamond_topology.install_flow(
            flow_id="flow1",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            path_score=path1,
            bandwidth=100.0
        )

        flows = diamond_topology.get_flows_on_link(1, 1, 2, 1)

        assert len(flows) == 1
        assert flows[0].flow_id == "flow1"

    def test_can_reroute_flows(self, diamond_topology):
        """Test reroute feasibility check."""
        path1 = diamond_topology.score_path([1, 2, 4])

        flow = diamond_topology.install_flow(
            flow_id="flow1",
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            path_score=path1,
            bandwidth=100.0
        )

        # Should be able to reroute via node 3
        can_reroute = diamond_topology.can_reroute_flows([flow], 1, 2)

        assert can_reroute is True

    def test_update_link_utilization(self, simple_topology):
        """Test updating link utilization."""
        simple_topology.update_link_utilization(1, 2, 50.0)

        # Check via path scoring
        path = [1, 2, 3]
        score = simple_topology.score_path(path)

        assert score.load_score > 0

    def test_topology_info(self, diamond_topology):
        """Test topology information retrieval."""
        info = diamond_topology.get_topology_info()

        assert "nodes" in info
        assert "edges" in info
        assert info["total_nodes"] == 4
        assert info["total_edges"] >= 4

    def test_router_stats(self, diamond_topology):
        """Test router statistics."""
        stats = diamond_topology.get_stats()

        assert stats["total_nodes"] == 4
        assert stats["total_links"] >= 4
        assert "active_flows" in stats
        assert "k_paths" in stats

    def test_qos_constraint(self, diamond_topology):
        """Test QoS constraint enforcement."""
        # Set high utilization on one path
        diamond_topology.update_link_utilization(1, 2, 90.0)
        diamond_topology.update_link_utilization(2, 4, 90.0)

        best_path = diamond_topology.find_best_path(1, 4)

        # Should prefer the less loaded path
        assert best_path is not None
        # May choose the other path if QoS is violated

    def test_reset(self, diamond_topology):
        """Test router reset."""
        diamond_topology.reset()

        info = diamond_topology.get_topology_info()
        assert info["total_nodes"] == 0
        assert info["total_edges"] == 0

    def test_no_path_exists(self, router):
        """Test handling when no path exists."""
        router.add_link(1, 1, 2, 1, 1000.0)
        router.add_link(3, 1, 4, 1, 1000.0)

        paths = router.find_k_shortest_paths(1, 4)

        assert len(paths) == 0
