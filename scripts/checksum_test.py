import morphyne as mp
import params_template
import unittest


class ChecksumTest(unittest.TestCase):
    def test(self):
        instance = mp.create_from_yaml(params_template.params_yaml)
        instance.set_non_coherent_stimulation_rate(5 / 800)
        instance.set_reward_rate(0.002)
        t_stop = 1000
        tick_result = instance.tick_until(t_stop)

        neuron_spike_checksum = tick_result.neuron_spikes.product(axis=1).sum()
        channel_spike_checksum = tick_result.out_channel_spikes.product(
            axis=1).sum()

        self.assertEqual(neuron_spike_checksum, 6761644859)
        self.assertEqual(channel_spike_checksum, 638290938)
        self.assertEqual(tick_result.synaptic_transmission_count, 4008419)

        instance.set_non_coherent_stimulation_rate(0.0)
        tick_result = instance.tick(extract_state_snapshot=True)

        voltage_checksum = tick_result.state_snapshot.membrane_voltages.sum()
        synapse_state_checksum = tick_result.state_snapshot.synapse_states.product(
            axis=1).sum()

        self.assertEqual(tick_result.spiking_nids.sum(), 17057)
        self.assertEqual(tick_result.spiking_out_channel_ids.sum(), 1857)
        self.assertEqual(tick_result.synaptic_transmission_count, 627)
        self.assertAlmostEqual(voltage_checksum, 102.19031620648457)
        self.assertAlmostEqual(synapse_state_checksum,
                               25466586780.556538, places=3)


if __name__ == '__main__':
    unittest.main()
