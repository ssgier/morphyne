import json
from typing import Optional

import numpy as np
import pandas as pd

from .morphyne import Stimulus
from .morphyne import create_from_json as create_from_json_inner
from .morphyne import create_from_yaml as create_from_yaml_inner


class StateSnapshot:
    def __init__(
        self, membrane_voltages: np.ndarray, synapse_states: pd.DataFrame
    ) -> None:
        self.membrane_voltages = membrane_voltages
        self.synapse_states = synapse_states


class TickResult:
    def __init__(
        self,
        out_channel_spikes: pd.DataFrame,
        neuron_spikes: pd.DataFrame,
        synaptic_transmission_count: int,
    ) -> None:
        self.out_channel_spikes = out_channel_spikes
        self.neuron_spikes = neuron_spikes
        self.synaptic_transmission_count = synaptic_transmission_count


def create_from_yaml(params_yaml: str, seed: int):
    if seed is None:
        seed = 0
    return Instance(create_from_yaml_inner(params_yaml, seed))


def create_from_json(params_json: str, seed=None):
    if seed is None:
        seed = 0
    return Instance(create_from_json_inner(params_json, seed))


def create(params: dict, seed=None):
    return create_from_json(json.dumps(params), seed)


def create_stimulus(
    in_channel_spikes: pd.DataFrame | None = None,
    force_out_channel_spikes: pd.DataFrame | None = None,
    force_neuron_spikes: pd.DataFrame | None = None,
) -> Stimulus:
    stimulus = Stimulus()

    if in_channel_spikes is not None:
        stimulus.in_channel_spikes_ts = in_channel_spikes.t.to_list()
        stimulus.in_channel_spikes_ids = in_channel_spikes.in_channel_id.to_list()

    if force_out_channel_spikes is not None:
        stimulus.force_out_channel_spikes_ts = force_out_channel_spikes.t.to_list()
        stimulus.force_out_channel_spikes_ids = (
            force_out_channel_spikes.out_channel_id.to_list()
        )

    if force_neuron_spikes is not None:
        stimulus.force_neuron_spikes_ts = force_neuron_spikes.t.to_list()
        stimulus.force_neuron_spikes_ids = force_neuron_spikes.nid.to_list()

    return stimulus


def extract_in_channel_spikes(stimulus: Stimulus) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": stimulus.in_channel_spikes_ts,
            "in_channel_id": stimulus.in_channel_spikes_ids,
        }
    )


def extract_force_out_channel_spikes(stimulus: Stimulus) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": stimulus.force_out_channel_spikes_ts,
            "out_channel_id": stimulus.force_out_channel_spikes_ids,
        }
    )


def extract_force_neuron_spikes(stimulus: Stimulus) -> pd.DataFrame:
    return pd.DataFrame(
        {"t": stimulus.force_neuron_spikes_ts, "nid": stimulus.force_neuron_spikes_ids}
    )


class Instance:
    def __init__(self, inner) -> None:
        self._inner = inner

    def apply_stimulus(self, stimulus: Stimulus):
        self._inner.apply_stimulus(stimulus)

    def tick(
        self,
        spiking_in_channel_ids=[],
        force_spiking_out_channel_ids=[],
        force_spiking_nids=[],
        reward=None,
        append_to: TickResult | None = None,
    ) -> TickResult:
        if reward is None:
            reward = self._inner.reward_rate

        inner_result = self._inner.tick(
            spiking_in_channel_ids,
            force_spiking_out_channel_ids,
            force_spiking_nids,
            reward,
        )

        spiking_out_channel_ids = np.array(inner_result.spiking_out_channel_ids)
        spiking_nids = np.array(inner_result.spiking_nids)

        neuron_spikes_data = {"t": inner_result.t, "nid": spiking_nids}
        out_channel_spikes_data = {
            "t": inner_result.t,
            "out_channel_id": spiking_out_channel_ids,
        }

        df_neuron_spikes = pd.DataFrame(neuron_spikes_data, dtype=np.int64)
        df_out_channel_spikes = pd.DataFrame(out_channel_spikes_data, dtype=np.int64)

        state_snapshot = None

        if append_to:
            return concat_results(
                append_to,
                df_out_channel_spikes,
                df_neuron_spikes,
                inner_result.synaptic_transmission_count,
            )
        else:
            return TickResult(
                df_out_channel_spikes,
                df_neuron_spikes,
                inner_result.synaptic_transmission_count,
            )

    def tick_until(self, t: int, append_to: TickResult | None = None) -> TickResult:
        inner_result = self._inner.tick_until(t)
        out_channel_spikes_data = {
            "t": inner_result.out_channel_spikes_ts,
            "out_channel_id": inner_result.out_channel_spikes_ids,
        }
        neuron_spikes_data = {
            "t": inner_result.neuron_spikes_ts,
            "nid": inner_result.neuron_spikes_ids,
        }

        df_out_channel_spikes = pd.DataFrame(out_channel_spikes_data, dtype=np.int64)
        df_neuron_spikes = pd.DataFrame(neuron_spikes_data, dtype=np.int64)

        if append_to:
            return concat_results(
                append_to,
                df_out_channel_spikes,
                df_neuron_spikes,
                inner_result.synaptic_transmission_count,
            )
        else:
            return TickResult(
                df_out_channel_spikes,
                df_neuron_spikes,
                inner_result.synaptic_transmission_count,
            )

    def tick_for(self, t: int, append_to: Optional[TickResult] = None) -> TickResult:
        return self.tick_until(self.get_next_t() + t, append_to=append_to)

    def extract_state_snapshot(self) -> StateSnapshot:
        inner_snapshot = self._inner.extract_state_snapshot()
        membrane_voltages = np.array(inner_snapshot.membrane_voltages)
        synapse_state_data = {
            "projection_id": inner_snapshot.projection_ids,
            "pre_syn_nid": inner_snapshot.pre_syn_nids,
            "post_syn_nid": inner_snapshot.post_syn_nids,
            "conduction_delay": inner_snapshot.conduction_delays,
            "weight": inner_snapshot.weights,
        }
        df_synapse_states = pd.DataFrame(synapse_state_data)
        state_snapshot = StateSnapshot(membrane_voltages, df_synapse_states)

        return state_snapshot

    def reset_ephemeral_state(self) -> None:
        self._inner.reset_ephemeral_state()

    def set_reward_rate(self, reward_rate):
        self._inner.reward_rate = reward_rate

    def set_non_coherent_stimulation_rate(self, rate):
        self._inner.non_coherent_stimulation_rate = rate

    def set_sc_off(self):
        self._inner.set_sc_off()

    def set_sc_single(self, threshold):
        self._inner.set_sc_single(threshold)

    def set_sc_multi(self, threshold):
        self._inner.set_sc_multi(threshold)

    def flush_sc_hashes(self) -> set[int]:
        return self._inner.flush_sc_hashes()

    def get_next_t(self) -> int:
        return self._inner.get_next_t()

    def get_last_t(self) -> int:
        return self._inner.get_last_t()

    def get_num_in_channels(self) -> int:
        return self._inner.get_num_in_channels()

    def get_num_out_channels(self) -> int:
        return self._inner.get_num_out_channels()

    def get_num_neurons(self) -> int:
        return self._inner.get_num_neurons()


def concat_results(
    result: TickResult,
    df_out_channel_spikes: pd.DataFrame,
    df_neuron_spikes: pd.DataFrame,
    syn_transmission_count,
) -> TickResult:
    df_out_channel_spikes = pd.concat(
        [result.out_channel_spikes, df_out_channel_spikes]
    )
    df_out_channel_spikes.reset_index(inplace=True, drop=True)

    df_neuron_spikes = pd.concat([result.neuron_spikes, df_neuron_spikes])
    df_neuron_spikes.reset_index(inplace=True, drop=True)

    return TickResult(df_out_channel_spikes, df_neuron_spikes, syn_transmission_count)
