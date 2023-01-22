import morphyne as mp
import pandas as pd
import matplotlib.pyplot as plt
import json

params = mp.get_example_params()
print(json.dumps(params, indent=4))

instance = mp.create(params)
instance.set_non_coherent_stimulation_rate(0.0005)
instance.set_reward_rate(0.1)

stimulus_df = pd.DataFrame({"t": [0, 5, 10], "in_channel_id": [0, 1, 2]})
stimulus = mp.create_stimulus(in_channel_spikes=stimulus_df)

instance.apply_stimulus(stimulus)
tick_result = instance.tick_until(1000)

plt.scatter(tick_result.neuron_spikes.t, tick_result.neuron_spikes.nid, s=1)
plt.xlabel("t")
plt.ylabel("neuron ID")
plt.show()
