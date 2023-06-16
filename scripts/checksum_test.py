import morphyne as mp
import params_template
import unittest


class ChecksumTest(unittest.TestCase):
    def test_vs_inner(self):
        instance = mp.create_from_yaml(params_template.params_yaml, seed=0)
        instance.set_non_coherent_stimulation_rate(0.01)
        instance.set_reward_rate(0.002)
        t_stop = 1001
        tick_result = instance.tick_until(t_stop)

        neuron_spike_checksum = tick_result.neuron_spikes.product(axis=1).sum()
        channel_spike_checksum = tick_result.out_channel_spikes.product(
            axis=1).sum()

        self.assertEqual(neuron_spike_checksum, 7857125832)
        self.assertEqual(channel_spike_checksum, 681382654)
        self.assertEqual(tick_result.synaptic_transmission_count, 3847971)

        instance.set_non_coherent_stimulation_rate(0.0)
        tick_result = instance.tick()
        state_snapshot = instance.extract_state_snapshot()

        self.check_single_tick_result(tick_result, state_snapshot)

    def check_single_tick_result(self, tick_result, state_snapshot):
        voltage_checksum = state_snapshot.membrane_voltages.sum()
        synapse_state_checksum = state_snapshot.synapse_states.drop(columns=["projection_id"], inplace=False).product(
            axis=1).sum()

        self.assertEqual(tick_result.neuron_spikes.nid.sum(), 15295)
        self.assertEqual(
            tick_result.out_channel_spikes.out_channel_id.sum(), 2495)
        self.assertEqual(tick_result.synaptic_transmission_count, 4022)
        self.assertAlmostEqual(voltage_checksum, 45.53336342307739)
        self.assertAlmostEqual(synapse_state_checksum,
                               87545291384.9079, places=2)


if __name__ == '__main__':
    unittest.main()
