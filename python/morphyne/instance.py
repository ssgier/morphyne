import numpy as np
import pandas as pd
from .morphyne import create_from_yaml as create_from_yaml_inner
from .morphyne import create_from_json as create_from_json_inner


class StateSnapshot:

    def __init__(self, membrane_voltages: np.array, synapse_states: pd.DataFrame) -> None:
        self.membrane_voltages = membrane_voltages
        self.synapse_states = synapse_states


class TickResult:

    def __init__(self, out_channel_spikes: pd.DataFrame, neuron_spikes: pd.DataFrame, state_snapshot: StateSnapshot, synaptic_transmission_count: int) -> None:
        self.out_channel_spikes = out_channel_spikes
        self.neuron_spikes = neuron_spikes
        self.state_snapshot = state_snapshot
        self.synaptic_transmission_count = synaptic_transmission_count


def create_from_yaml(params_yaml: str):
    return Instance(create_from_yaml_inner(params_yaml))


def create_from_json(params_json: str):
    return Instance(create_from_json_inner(params_json))


class Instance:

    def __init__(self, inner) -> None:
        self._inner = inner

    def tick(self, spiking_in_channel_ids=[], force_spiking_out_channel_ids=[], force_spiking_nids=[], reward=None, extract_state_snapshot=False) -> TickResult:

        if reward is None:
            reward = self._inner.reward_rate

        inner_result = self._inner.tick(
            spiking_in_channel_ids, force_spiking_out_channel_ids, force_spiking_nids, reward, extract_state_snapshot)

        spiking_out_channel_ids = np.array(
            inner_result.spiking_out_channel_ids)
        spiking_nids = np.array(inner_result.spiking_nids)

        neuron_spikes_data = {"t": inner_result.t, "nid": spiking_nids}
        out_channel_spikes_data = {"t": inner_result.t,
                                   "out_channel_id": spiking_out_channel_ids}

        df_neuron_spikes = pd.DataFrame(neuron_spikes_data)
        df_out_channel_spikes = pd.DataFrame(out_channel_spikes_data)

        state_snapshot = None
        if extract_state_snapshot:
            membrane_voltages = np.array(
                inner_result.state_snapshot.membrane_voltages)
            synapse_state_data = {"pre_syn_nid": inner_result.state_snapshot.pre_syn_nids,
                                  "post_syn_nid": inner_result.state_snapshot.post_syn_nids,
                                  "conduction_delay": inner_result.state_snapshot.conduction_delays,
                                  "weight": inner_result.state_snapshot.weights}
            df_synapse_states = pd.DataFrame(synapse_state_data)
            state_snapshot = StateSnapshot(
                membrane_voltages, df_synapse_states)

        return TickResult(df_out_channel_spikes, df_neuron_spikes, state_snapshot, inner_result.synaptic_transmission_count)

    def tick_until(self, t: int, ignore_output=False) -> TickResult:
        if ignore_output:
            self._inner.tick_until(t)
            return None
        else:
            inner_result = self._inner.tick_until(t)
            out_channel_spikes_data = {"t": inner_result.out_channel_spikes_ts,
                                       "out_channel_id": inner_result.out_channel_spikes_ids}
            neuron_spikes_data = {
                "t": inner_result.neuron_spikes_ts, "nid": inner_result.neuron_spikes_ids}
            return TickResult(pd.DataFrame(out_channel_spikes_data), pd.DataFrame(neuron_spikes_data), None, inner_result.synaptic_transmission_count)

    def set_reward_rate(self, reward_rate):
        self._inner.reward_rate = reward_rate

    def set_non_coherent_stimulation_rate(self, rate):
        self._inner.non_coherent_stimulation_rate = rate

    def get_t(self) -> int:
        return self._inner.get_t()
