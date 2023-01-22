import morphyne as mp
import time
import params_template

instance = mp.create_from_yaml(params_template.params_yaml, 0)
instance.set_non_coherent_stimulation_rate(5 / 800)

synaptic_transmission_count = 0
spike_count = 0
t_stop = 50000
batch_size = 1000

t_start = time.time()

while instance.get_t() < t_stop:
    tick_result = instance.tick_until(
        min(t_stop, instance.get_t() + batch_size))
    synaptic_transmission_count += tick_result.synaptic_transmission_count
    spike_count += len(tick_result.neuron_spikes)

t_end = time.time()

synaptic_transmission_processing_throughput = synaptic_transmission_count / \
    (t_end - t_start)

print("Spikes per cycle: {}".format(spike_count / t_stop))
print("Synaptic transmission processing throughput: {:.3e}, ({:.2f} ns per transmission)".format(
    synaptic_transmission_processing_throughput, 1e9 / synaptic_transmission_processing_throughput))
