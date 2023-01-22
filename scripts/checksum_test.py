import morphyne as mp
import params_template
import unittest


class ChecksumTest(unittest.TestCase):
    def test_vs_inner(self):
        instance = mp.create_from_yaml(params_template.params_yaml, seed=0)
        instance.set_non_coherent_stimulation_rate(0.01)
        instance.set_reward_rate(0.002)
        t_stop = 1000
        tick_result = instance.tick_until(t_stop)

        neuron_spike_checksum = tick_result.neuron_spikes.product(axis=1).sum()
        channel_spike_checksum = tick_result.out_channel_spikes.product(
            axis=1).sum()

        self.assertEqual(neuron_spike_checksum, 9325888458)
        self.assertEqual(channel_spike_checksum, 857971287)
        self.assertEqual(tick_result.synaptic_transmission_count, 5456365)

        instance.set_non_coherent_stimulation_rate(0.0)
        tick_result = instance.tick(extract_state_snapshot=True)

        self.check_single_tick_result(tick_result)

    def test_ignore_output(self):
        instance = mp.create_from_yaml(params_template.params_yaml, seed=0)
        instance.set_non_coherent_stimulation_rate(0.01)
        instance.set_reward_rate(0.002)
        t_stop = 1000
        instance.tick_until(t_stop, ignore_output=True)
        instance.set_non_coherent_stimulation_rate(0.0)
        tick_result = instance.tick(extract_state_snapshot=True)
        self.check_single_tick_result(tick_result)

    def check_single_tick_result(self, tick_result):
        voltage_checksum = tick_result.state_snapshot.membrane_voltages.sum()
        synapse_state_checksum = tick_result.state_snapshot.synapse_states.product(
            axis=1).sum()

        self.assertEqual(tick_result.neuron_spikes.nid.sum(), 39115)
        self.assertEqual(
            tick_result.out_channel_spikes.out_channel_id.sum(), 4715)
        self.assertEqual(tick_result.synaptic_transmission_count, 1525)
        self.assertAlmostEqual(voltage_checksum, 79.48388550709933)
        self.assertAlmostEqual(synapse_state_checksum,
                               94369097433.76978, places=2)


if __name__ == '__main__':
    unittest.main()
