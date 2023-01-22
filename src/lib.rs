use morphine::{
    instance::{self, TickInput},
    params::InstanceParams,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{prelude::Distribution, rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::de::Error;
use statrs::distribution::Poisson;
use std::{collections::VecDeque, iter};

struct SpikeInfo {
    t: usize,
    id: usize,
}

impl SpikeInfo {
    fn new(t: usize, id: usize) -> Self {
        Self { t, id }
    }
}

#[pyclass]
struct Instance {
    inner: instance::Instance,
    rng: StdRng,
    non_coherent_stimulation_nids: Vec<usize>,

    #[pyo3(get, set)]
    reward_rate: f32,

    #[pyo3(get, set)]
    non_coherent_stimulation_rate: f64,

    in_channel_stimuli: Vec<VecDeque<SpikeInfo>>,

    force_out_channel_stimuli: Vec<VecDeque<SpikeInfo>>,

    force_neuron_stimuli: Vec<VecDeque<SpikeInfo>>,
}

#[pyclass]
#[derive(Debug)]
struct Stimulus {
    #[pyo3(get, set)]
    in_channel_spikes_ts: Vec<usize>,

    #[pyo3(get, set)]
    in_channel_spikes_ids: Vec<usize>,

    #[pyo3(get, set)]
    force_out_channel_spikes_ts: Vec<usize>,

    #[pyo3(get, set)]
    force_out_channel_spikes_ids: Vec<usize>,

    #[pyo3(get, set)]
    force_neuron_spikes_ts: Vec<usize>,

    #[pyo3(get, set)]
    force_neuron_spikes_ids: Vec<usize>,
}

#[pyclass]
struct TickResult {
    #[pyo3(get)]
    t: usize,

    #[pyo3(get)]
    spiking_out_channel_ids: Vec<usize>,

    #[pyo3(get)]
    spiking_nids: Vec<usize>,

    #[pyo3(get)]
    synaptic_transmission_count: usize,

    #[pyo3(get)]
    state_snapshot: Option<StateSnapshot>,
}

#[pyclass]
struct BatchTickResult {
    #[pyo3(get)]
    out_channel_spikes_ts: Vec<usize>,

    #[pyo3(get)]
    out_channel_spikes_ids: Vec<usize>,

    #[pyo3(get)]
    neuron_spikes_ts: Vec<usize>,

    #[pyo3(get)]
    neuron_spikes_ids: Vec<usize>,

    #[pyo3(get)]
    synaptic_transmission_count: usize,
}

#[pyclass]
#[derive(Clone)]
struct StateSnapshot {
    #[pyo3(get)]
    membrane_voltages: Vec<f32>,

    #[pyo3(get)]
    pre_syn_nids: Vec<usize>,

    #[pyo3(get)]
    post_syn_nids: Vec<usize>,

    #[pyo3(get)]
    conduction_delays: Vec<u8>,

    #[pyo3(get)]
    weights: Vec<f32>,
}

#[pymethods]
impl Instance {
    fn tick(
        &mut self,
        spiking_in_channel_ids: Vec<usize>,
        force_spiking_out_channel_ids: Vec<usize>,
        force_spiking_nids: Vec<usize>,
        reward: f32,
        extract_state_snapshot: bool,
    ) -> PyResult<TickResult> {
        let mut tick_input = TickInput {
            spiking_in_channel_ids,
            force_spiking_out_channel_ids,
            force_spiking_nids,
            reward,
            extract_state_snapshot,
        };

        self.add_stimulation(&mut tick_input);

        let inner_result = self
            .inner
            .tick(&tick_input)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;

        let state_snapshot = if extract_state_snapshot {
            let inner_snapshot = inner_result.state_snapshot.unwrap();

            let membrane_voltages = inner_snapshot
                .neuron_states
                .into_iter()
                .map(|neuron_state| neuron_state.voltage)
                .collect();
            let pre_syn_nids = inner_snapshot
                .synapse_states
                .iter()
                .map(|syn_state| syn_state.pre_syn_nid)
                .collect();
            let post_syn_nids = inner_snapshot
                .synapse_states
                .iter()
                .map(|syn_state| syn_state.post_syn_nid)
                .collect();
            let conduction_delays = inner_snapshot
                .synapse_states
                .iter()
                .map(|syn_state| syn_state.conduction_delay)
                .collect();
            let weights = inner_snapshot
                .synapse_states
                .iter()
                .map(|syn_state| syn_state.weight)
                .collect();

            Some(StateSnapshot {
                membrane_voltages,
                pre_syn_nids,
                post_syn_nids,
                conduction_delays,
                weights,
            })
        } else {
            None
        };

        Ok(TickResult {
            t: inner_result.t,
            spiking_nids: inner_result.spiking_nids,
            spiking_out_channel_ids: inner_result.spiking_out_channel_ids,
            synaptic_transmission_count: inner_result.synaptic_transmission_count,
            state_snapshot,
        })
    }

    fn tick_until(&mut self, t: usize) -> BatchTickResult {
        let mut out_channel_spikes_ts = Vec::new();
        let mut out_channel_spikes_ids = Vec::new();
        let mut neuron_spikes_ts = Vec::new();
        let mut neuron_spikes_ids = Vec::new();
        let mut synaptic_transmission_count = 0;

        let mut tick_input = TickInput::new();

        while self.inner.get_tick_period() < t {
            tick_input.reset();
            tick_input.reward = self.reward_rate;
            self.add_stimulation(&mut tick_input);

            let tick_result = self.inner.tick(&tick_input).unwrap();

            out_channel_spikes_ts.extend(
                iter::repeat(tick_result.t).take(tick_result.spiking_out_channel_ids.len()),
            );
            out_channel_spikes_ids.extend(tick_result.spiking_out_channel_ids);

            neuron_spikes_ts
                .extend(iter::repeat(tick_result.t).take(tick_result.spiking_nids.len()));
            neuron_spikes_ids.extend(tick_result.spiking_nids);

            synaptic_transmission_count += tick_result.synaptic_transmission_count;
        }

        BatchTickResult {
            out_channel_spikes_ts,
            out_channel_spikes_ids,
            neuron_spikes_ts,
            neuron_spikes_ids,
            synaptic_transmission_count,
        }
    }

    fn tick_until_ignore_output(&mut self, t: usize) {
        let mut tick_input = TickInput::new();

        while self.inner.get_tick_period() < t {
            tick_input.reset();
            tick_input.reward = self.reward_rate;
            self.add_stimulation(&mut tick_input);
            self.inner.tick(&tick_input).unwrap();
        }
    }

    fn apply_stimulus(&mut self, stimulus: &Stimulus) {
        if !stimulus.in_channel_spikes_ts.is_empty() {
            let in_channel_stimulus: VecDeque<_> = stimulus
                .in_channel_spikes_ts
                .iter()
                .map(|t| t + self.get_t())
                .zip(stimulus.in_channel_spikes_ids.iter())
                .map(|(t, id)| SpikeInfo::new(t, *id))
                .collect();

            self.in_channel_stimuli.push(in_channel_stimulus);
        }

        if !stimulus.force_out_channel_spikes_ts.is_empty() {
            let force_out_channel_stimulus: VecDeque<_> = stimulus
                .force_out_channel_spikes_ts
                .iter()
                .map(|t| t + self.get_t())
                .zip(stimulus.force_out_channel_spikes_ids.iter())
                .map(|(t, id)| SpikeInfo::new(t, *id))
                .collect();

            self.force_out_channel_stimuli
                .push(force_out_channel_stimulus);
        }

        if !stimulus.force_neuron_spikes_ts.is_empty() {
            let force_neuron_stimulus: VecDeque<_> = stimulus
                .force_neuron_spikes_ts
                .iter()
                .map(|t| t + self.get_t())
                .zip(stimulus.force_neuron_spikes_ids.iter())
                .map(|(t, id)| SpikeInfo::new(t, *id))
                .collect();

            self.force_neuron_stimuli.push(force_neuron_stimulus);
        }
    }

    fn get_t(&self) -> usize {
        self.inner.get_tick_period()
    }
}

impl Instance {
    fn add_stimulation(&mut self, tick_input: &mut TickInput) {
        if self.non_coherent_stimulation_rate > 0.0 {
            let num_stimulus_spikes_dist = Poisson::new(
                self.non_coherent_stimulation_rate
                    * self.non_coherent_stimulation_nids.len() as f64,
            )
            .unwrap();

            let num_spikes = num_stimulus_spikes_dist.sample(&mut self.rng) as usize;

            tick_input.force_spiking_nids.extend(
                self.non_coherent_stimulation_nids
                    .choose_multiple(&mut self.rng, num_spikes)
                    .copied(),
            )
        }

        self.poll_stimulus_queues(tick_input);
    }

    fn poll_stimulus_queues(&mut self, tick_input: &mut TickInput) {
        for in_channel_stimulus in self.in_channel_stimuli.iter_mut() {
            while let Some(spike) = in_channel_stimulus.front() {
                if spike.t == self.inner.get_tick_period() {
                    tick_input.spiking_in_channel_ids.push(spike.id);
                } else {
                    break;
                }

                in_channel_stimulus.pop_front();
            }
        }

        self.in_channel_stimuli
            .retain(|stimulus| !stimulus.is_empty());

        for force_out_channel_stimulus in self.force_out_channel_stimuli.iter_mut() {
            while let Some(spike) = force_out_channel_stimulus.front() {
                if spike.t == self.inner.get_tick_period() {
                    tick_input.force_spiking_out_channel_ids.push(spike.id);
                } else {
                    break;
                }

                force_out_channel_stimulus.pop_front();
            }
        }

        self.force_out_channel_stimuli
            .retain(|stimulus| !stimulus.is_empty());

        for force_neuron_stimulus in self.force_neuron_stimuli.iter_mut() {
            while let Some(spike) = force_neuron_stimulus.front() {
                if spike.t == self.inner.get_tick_period() {
                    tick_input.force_spiking_nids.push(spike.id);
                } else {
                    break;
                }

                force_neuron_stimulus.pop_front();
            }
        }

        self.force_neuron_stimuli
            .retain(|stimulus| !stimulus.is_empty());
    }
}

#[pymethods]
impl Stimulus {
    #[new]
    fn new() -> Self {
        Self {
            in_channel_spikes_ts: Vec::new(),
            in_channel_spikes_ids: Vec::new(),
            force_out_channel_spikes_ts: Vec::new(),
            force_out_channel_spikes_ids: Vec::new(),
            force_neuron_spikes_ts: Vec::new(),
            force_neuron_spikes_ids: Vec::new(),
        }
    }
}

#[pyfunction]
fn create_from_yaml(yaml_str: &str, seed: u64) -> PyResult<Instance> {
    create_from_deser_result(serde_yaml::from_str(yaml_str), seed)
}

#[pyfunction]
fn create_from_json(json_str: &str, seed: u64) -> PyResult<Instance> {
    create_from_deser_result(serde_json::from_str(json_str), seed)
}

fn create_from_deser_result<E: Error>(
    result: Result<InstanceParams, E>,
    seed: u64,
) -> PyResult<Instance> {
    let params = result.map_err(|error| PyValueError::new_err(error.to_string()))?;

    let instance = instance::create_instance(params)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let non_coherent_stimulation_nids = (0..(instance.get_num_neurons())).collect::<Vec<_>>();

    Ok(Instance {
        inner: instance,
        reward_rate: 0.0,
        non_coherent_stimulation_rate: 0.0,
        non_coherent_stimulation_nids,
        rng: StdRng::seed_from_u64(seed),
        in_channel_stimuli: Vec::new(),
        force_out_channel_stimuli: Vec::new(),
        force_neuron_stimuli: Vec::new(),
    })
}

#[pymodule]
fn morphyne(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_from_yaml, m)?)?;
    m.add_function(wrap_pyfunction!(create_from_json, m)?)?;
    m.add_class::<Stimulus>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use morphine::{
        instance::{self},
        params::{InstanceParams, LayerParams},
    };
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{Instance, Stimulus};

    #[test]
    fn stimulus_queues_clean_up() {
        let mut params = InstanceParams::default();
        let layer_params = LayerParams::default();
        params.layers.push(layer_params);

        let mut stimulus = Stimulus::new();
        stimulus.in_channel_spikes_ts.push(1);
        stimulus.in_channel_spikes_ids.push(0);
        stimulus.force_out_channel_spikes_ts.push(1);
        stimulus.force_out_channel_spikes_ids.push(0);
        stimulus.force_neuron_spikes_ts.push(1);
        stimulus.force_neuron_spikes_ids.push(0);

        let mut instance = Instance {
            inner: instance::create_instance(params).unwrap(),
            reward_rate: 0.0,
            non_coherent_stimulation_rate: 0.0,
            non_coherent_stimulation_nids: Vec::new(),
            rng: StdRng::seed_from_u64(0),
            in_channel_stimuli: Vec::new(),
            force_out_channel_stimuli: Vec::new(),
            force_neuron_stimuli: Vec::new(),
        };

        instance.apply_stimulus(&stimulus);

        assert_eq!(instance.in_channel_stimuli.len(), 1);
        assert_eq!(instance.force_out_channel_stimuli.len(), 1);
        assert_eq!(instance.force_neuron_stimuli.len(), 1);

        instance.tick_until(1);

        assert_eq!(instance.in_channel_stimuli.len(), 1);
        assert_eq!(instance.force_out_channel_stimuli.len(), 1);
        assert_eq!(instance.force_neuron_stimuli.len(), 1);

        instance.tick_until(2);

        assert!(instance.in_channel_stimuli.is_empty());
        assert!(instance.force_out_channel_stimuli.is_empty());
        assert!(instance.force_neuron_stimuli.is_empty());
    }
}
