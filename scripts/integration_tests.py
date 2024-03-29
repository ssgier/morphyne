import morphyne as mp
import unittest
import json
import pandas as pd
import numpy as np
import random
from pandas.testing import assert_frame_equal


class IntegrationTests(unittest.TestCase):

    def test_stimulus_random(self):
        random.seed(0)
        np.random.seed(0)
        for _ in 0, 10:
            instance = create_instance()
            t_start = random.randint(0, 100)
            instance.tick_until(t_start)

            num_stimuli = random.randint(5, 10)

            stimuli = [generate_random_stimulus()
                       for _ in range(0, num_stimuli)]

            result = None
            expected_neuron_spikes = []

            for stimulus in stimuli:
                instance.apply_stimulus(stimulus)
                expected_neuron_spikes.append(
                    get_expected_neuron_spikes(stimulus, instance.get_next_t()))
                result = instance.tick(append_to=result)

            result = instance.tick_for(20, append_to=result)

            expected_neuron_spikes = pd.concat(expected_neuron_spikes)
            expected_neuron_spikes.sort_values(by=["t", "nid"], inplace=True)
            expected_neuron_spikes.reset_index(inplace=True, drop=True)

            expected_out_channel_spikes = extract_expected_out_channel_spikes(
                expected_neuron_spikes)

            assert_frame_equal(result.out_channel_spikes,
                               expected_out_channel_spikes)

            assert_frame_equal(result.neuron_spikes, expected_neuron_spikes)

    def test_stimulus_specific_single_ticks(self):
        instance = prepare_instance()
        result = None

        while instance.get_last_t() < 110:
            result = instance.tick(append_to=result)
        self.check_result(result)

    def test_stimulus_specific_batch(self):
        instance = prepare_instance()
        result = instance.tick_until(110)
        self.check_result(result)

    def test_unsorted_spikes(self):
        instance = create_instance()

        bad_stimuli = [
            mp.create_stimulus(in_channel_spikes=pd.DataFrame(
                {"t": [2, 1], "in_channel_id": [0, 0]})),
            mp.create_stimulus(force_out_channel_spikes=pd.DataFrame(
                {"t": [2, 1], "out_channel_id": [0, 0]})),
            mp.create_stimulus(force_neuron_spikes=pd.DataFrame(
                {"t": [2, 1], "nid": [0, 0]})),
        ]

        for bad_stimulus in bad_stimuli:
            with self.assertRaises(ValueError):
                instance.apply_stimulus(bad_stimulus)

    def check_result(self, result: pd.DataFrame):
        expected_out_channel_spikes = pd.DataFrame(
            {"t": [103, 104, 104, 108], "out_channel_id": [0, 2, 2, 1]})

        expected_neuron_spikes = pd.DataFrame(
            {"t": [102, 103, 103, 103, 103, 104, 104, 104, 105, 108, 109],
                "nid": [3, 1, 1, 2, 11, 7, 13, 13, 2, 12, 8]}
        )

        assert_frame_equal(result.out_channel_spikes,
                           expected_out_channel_spikes)

        assert_frame_equal(result.neuron_spikes, expected_neuron_spikes)


def get_expected_neuron_spikes(stimulus: mp.Stimulus, t_application: int) -> pd.DataFrame:
    df_in_channel_spikes = mp.extract_in_channel_spikes(stimulus)
    df_force_out_channel_spikes = mp.extract_force_out_channel_spikes(stimulus)
    df_force_neuron_spikes = mp.extract_force_neuron_spikes(stimulus)

    df_in_channel_spikes["nid"] = df_in_channel_spikes["in_channel_id"]
    df_force_out_channel_spikes["nid"] = df_force_out_channel_spikes["out_channel_id"] + 11

    df_in_channel_spikes.drop(columns=["in_channel_id"], inplace=True)
    df_force_out_channel_spikes.drop(columns=["out_channel_id"], inplace=True)

    df = pd.concat(
        [df_in_channel_spikes, df_force_out_channel_spikes, df_force_neuron_spikes])

    df["t"] = df["t"] + t_application

    return df


def extract_expected_out_channel_spikes(expected_neuron_spikes: pd.DataFrame) -> pd.DataFrame:
    filtered_df = expected_neuron_spikes.loc[expected_neuron_spikes.nid >= 11].copy(
    )
    filtered_df["out_channel_id"] = filtered_df.nid - 11
    filtered_df.drop(columns=["nid"], inplace=True)
    filtered_df.reset_index(inplace=True, drop=True)
    return filtered_df


def generate_random_stimulus() -> mp.Stimulus:
    t = 20
    in_channel_spikes = generate_random_spike_train(t, 6, 6, "in_channel_id")
    force_out_channel_spikes = generate_random_spike_train(
        t, 3, 3, "out_channel_id")
    force_neuron_spikes = generate_random_spike_train(t, 14, 14, "nid")

    stimulus = mp.create_stimulus(in_channel_spikes=in_channel_spikes,
                                  force_out_channel_spikes=force_out_channel_spikes, force_neuron_spikes=force_neuron_spikes)
    return stimulus


def generate_random_spike_train(t: int, num_ids: int, max_size: int, id_col_name: str) -> pd.DataFrame:
    size = random.randint(1, max_size)
    ts = np.random.randint(1, t, size=size)
    ids = np.random.randint(num_ids, size=size)
    data = {"t": ts, id_col_name: ids}
    df = pd.DataFrame(data)
    df.sort_values(by="t", inplace=True)
    return df


def prepare_instance() -> mp.Instance:
    instance = create_instance()
    instance.tick_until(100)

    in_channel_spikes = pd.DataFrame(
        {"t": [3, 3, 5], "in_channel_id": [1, 2, 2]})
    force_out_channel_spikes = pd.DataFrame(
        {"t": [3, 4], "out_channel_id": [0, 2]})
    force_neuron_spikes = pd.DataFrame({"t": [3, 4, 4], "nid": [1, 7, 13]})

    stimulus_0 = mp.create_stimulus(in_channel_spikes=in_channel_spikes,
                                    force_out_channel_spikes=force_out_channel_spikes, force_neuron_spikes=force_neuron_spikes)

    in_channel_spikes = pd.DataFrame(
        {"t": [1], "in_channel_id": [3]})
    force_out_channel_spikes = pd.DataFrame(
        {"t": [7], "out_channel_id": [1]})
    force_neuron_spikes = pd.DataFrame({"t": [8], "nid": [8]})

    stimulus_1 = mp.create_stimulus(in_channel_spikes=in_channel_spikes,
                                    force_out_channel_spikes=force_out_channel_spikes, force_neuron_spikes=force_neuron_spikes)

    instance.apply_stimulus(stimulus_0)
    instance.tick()
    instance.apply_stimulus(stimulus_1)

    return instance


def create_instance() -> mp.Instance:
    mp_params = {
        "position_dim": 0,
        "hyper_sphere": False,
        "layers": [
            {
                "num_neurons": 6,
                "neuron_params": {
                    "tau_membrane": 10,
                    "refractory_period": 1,
                    "reset_voltage": 0,
                    "t_cutoff_coincidence": 20,
                    "adaptation_threshold": 1,
                    "tau_threshold": 50,
                    "voltage_floor": 0
                },
                "use_para_spikes": False
            },
            {
                "num_neurons": 5,
                "neuron_params": {
                    "tau_membrane": 10,
                    "refractory_period": 1,
                    "reset_voltage": 0,
                    "t_cutoff_coincidence": 20,
                    "adaptation_threshold": 1,
                    "tau_threshold": 50,
                    "voltage_floor": 0
                },
                "use_para_spikes": False
            },
            {
                "num_neurons": 3,
                "neuron_params": {
                    "tau_membrane": 10,
                    "refractory_period": 1,
                    "reset_voltage": 0,
                    "t_cutoff_coincidence": 20,
                    "adaptation_threshold": 1,
                    "tau_threshold": 50,
                    "voltage_floor": 0
                },
                "use_para_spikes": False
            }
        ],
        "projections": [
        ],
        "technical_params": {
            "num_threads": 1,
            "pin_threads": False
        }
    }

    instance = mp.create_from_json(json.dumps(mp_params))
    return instance


if __name__ == "__main__":
    unittest.main()
