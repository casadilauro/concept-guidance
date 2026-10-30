import unittest

import numpy as np
from sklearn.preprocessing import normalize

from recipe_diversity.metrics import (
    covering_curve,
    covering_number,
    manifold_coverage,
    rho_at,
    rho_curve,
    scaffold_levels,
    stratification,
)


def clustered(groups: int, per_group: int, spread: float, seed: int = 0) -> np.ndarray:
    """`groups` well-separated blobs, each `spread` wide, on the unit sphere."""
    generator = np.random.default_rng(seed)
    centers = normalize(generator.normal(size=(groups, 64)))
    points = np.repeat(centers, per_group, axis=0)
    return normalize(points + spread * generator.normal(size=points.shape))


class RhoCurveTests(unittest.TestCase):
    def test_matches_the_hand_computed_value_at_a_small_k(self) -> None:
        vectors = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)

        np.testing.assert_allclose(rho_at(vectors, 1), np.array([1.0, 1.0, 4.0]))
        np.testing.assert_allclose(rho_at(vectors, 2), np.array([5.0, 2.5, 6.5]))

    def test_omits_k_values_the_population_cannot_support(self) -> None:
        curve = rho_curve(np.random.default_rng(0).normal(size=(12, 4)), ks=(1, 5, 11, 15, 50))

        self.assertEqual([row["k"] for row in curve], [1, 5, 11])

    def test_rejects_a_k_larger_than_the_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs at least 4 vectors"):
            rho_at(np.zeros((3, 2)), 3)

    def test_reports_the_crossover_that_a_single_k_hides(self) -> None:
        """Tight clusters lose locally and win globally; one k cannot say both."""
        grouped = clustered(groups=10, per_group=10, spread=0.05)
        even = normalize(np.random.default_rng(1).normal(size=(100, 64)))

        local = rho_at(grouped, 3).mean() / rho_at(even, 3).mean()
        globalish = rho_at(grouped, 50).mean() / rho_at(even, 50).mean()

        self.assertLess(local, 1.0)
        self.assertGreater(globalish, local)


class CoveringTests(unittest.TestCase):
    def test_counts_one_mode_per_separated_cluster(self) -> None:
        vectors = clustered(groups=6, per_group=20, spread=0.01)

        self.assertEqual(covering_number(vectors, 0.5), 6)

    def test_a_wider_radius_never_needs_more_centers(self) -> None:
        vectors = clustered(groups=8, per_group=10, spread=0.1)
        counts = [row["modes"] for row in covering_curve(vectors, (0.05, 0.2, 0.5, 0.9))]

        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_handles_an_empty_population(self) -> None:
        self.assertEqual(covering_number(np.zeros((0, 4)), 0.3), 0)


class StratificationTests(unittest.TestCase):
    def test_real_groups_score_far_above_the_shuffled_null(self) -> None:
        vectors = clustered(groups=10, per_group=10, spread=0.05)
        labels = [f"g{index // 10}" for index in range(100)]
        result = stratification(vectors, labels, repeats=5)

        self.assertEqual(result["groups"], 10)
        self.assertGreater(result["ratio_to_null"], 10.0)
        self.assertGreater(result["between_within"], result["null_between_within"])

    def test_labels_unrelated_to_the_geometry_score_at_the_null(self) -> None:
        vectors = normalize(np.random.default_rng(2).normal(size=(100, 64)))
        labels = [f"g{index % 10}" for index in range(100)]
        result = stratification(vectors, labels, repeats=5)

        self.assertLess(result["ratio_to_null"], 3.0)

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        vectors = clustered(groups=5, per_group=8, spread=0.1)
        labels = [f"g{index // 8}" for index in range(40)]

        self.assertEqual(
            stratification(vectors, labels, seed=7),
            stratification(vectors, labels, seed=7),
        )

    def test_rejects_mismatched_labels_and_single_group_input(self) -> None:
        vectors = clustered(groups=2, per_group=5, spread=0.1)
        with self.assertRaisesRegex(ValueError, "one label per vector"):
            stratification(vectors, ["a"] * 3)
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            stratification(vectors, ["a"] * 10)


class ManifoldCoverageTests(unittest.TestCase):
    def test_a_population_sitting_on_the_reference_covers_all_of_it(self) -> None:
        reference = clustered(groups=8, per_group=10, spread=0.05, seed=3)
        result = manifold_coverage(reference, reference)

        self.assertEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(result["median_distance"], 0.0, places=6)

    def test_a_population_reaching_only_some_clusters_covers_only_those(self) -> None:
        reference = clustered(groups=10, per_group=10, spread=0.02, seed=4)
        result = manifold_coverage(reference, reference[:30])

        self.assertGreater(result["coverage"], 0.2)
        self.assertLess(result["coverage"], 0.5)

    def test_dispersion_alone_does_not_produce_coverage(self) -> None:
        """A spread-out population can still miss the reference entirely."""
        reference = clustered(groups=5, per_group=20, spread=0.02, seed=5)
        elsewhere = normalize(np.random.default_rng(6).normal(size=(100, 64)))

        self.assertLess(manifold_coverage(reference, elsewhere)["coverage"], 0.2)

    def test_rejects_a_reference_too_small_for_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs at least 6 reference vectors"):
            manifold_coverage(np.zeros((4, 3)), np.zeros((10, 3)))


class ScaffoldLevelTests(unittest.TestCase):
    def test_reports_only_levels_present_on_every_record(self) -> None:
        metadata = [
            {"cuisine_id": "cuisine_000", "dish_id": "dish_000_000"},
            {"cuisine_id": "cuisine_001"},
        ]

        self.assertEqual(list(scaffold_levels(metadata)), ["cuisine_id"])
        self.assertEqual(scaffold_levels([{"strategy": "nonce"}]), {})
        self.assertEqual(scaffold_levels([]), {})

    def test_orders_levels_coarsest_first(self) -> None:
        metadata = [{"cuisine_id": "c", "dish_id": "d"}]

        self.assertEqual(list(scaffold_levels(metadata)), ["cuisine_id", "dish_id"])
