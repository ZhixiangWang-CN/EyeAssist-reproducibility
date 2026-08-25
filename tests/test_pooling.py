import numpy as np
import unittest

from eyeassist.pooling import held_out_configuration_scores


class PoolingTests(unittest.TestCase):
    def test_configuration_enumeration_excludes_target_and_recovers_match(self):
        state = {**{f"A{i}": "A" for i in range(5)}, **{f"B{i}": "B" for i in range(5)}}
        a = np.array([[0.90, 0.05], [0.03, 0.02]])
        b = np.array([[0.02, 0.03], [0.05, 0.90]])
        maps = {reader: (a if group == "A" else b) for reader, group in state.items()}
        target = a
        result = held_out_configuration_scores(
            maps=maps,
            target_reader="A0",
            state_by_reader=state,
            score=lambda pool: float(np.sum(target * np.log(pool))),
            matched_size=4,
            half_mixed_target=2,
            half_mixed_off_state=2,
            opposite_size=4,
        )
        self.assertGreater(result["matched"], result["half_mixed"])
        self.assertGreater(result["half_mixed"], result["opposite"])
        self.assertGreater(result["matched_minus_opposite"], 0)

    def test_session_axis_excludes_target_identity_across_both_sessions(self):
        records = [f"s{session}:r{reader}" for session in (1, 2) for reader in (1, 2, 3)]
        state = {record: record.split(":")[0] for record in records}
        identity = {record: record.split(":")[1] for record in records}
        maps = {}
        for index, record in enumerate(records):
            value = np.zeros(6, dtype=float)
            value[index] = 1.0
            maps[record] = value

        observed_pools = []

        def score(pool):
            observed_pools.append(pool.copy())
            return 0.0

        held_out_configuration_scores(
            maps=maps,
            target_reader="s1:r1",
            state_by_reader=state,
            identity_by_reader=identity,
            score=score,
            matched_size=2,
            half_mixed_target=1,
            half_mixed_off_state=1,
            opposite_size=2,
        )

        # Index 3 is the opposite-session record from target identity r1.
        self.assertTrue(all(pool[3] == 0 for pool in observed_pools))
        # Half-mixed pools must not combine both sessions from the same reader.
        invalid_pairs = ({1, 4}, {2, 5})
        for pool in observed_pools:
            active = {index for index, value in enumerate(pool) if value > 0}
            self.assertNotIn(active, invalid_pairs)
