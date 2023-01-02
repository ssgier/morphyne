use morphine::instance::{self, TickInput};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use std::iter;

#[pyclass]
struct Instance {
    inner: instance::Instance,
    rng: StdRng,
    non_coherent_stimulation_nids: Vec<usize>,

    #[pyo3(get, set)]
    reward_rate: f32,

    #[pyo3(get, set)]
    non_coherent_stimulation_rate: f64,
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

        self.add_non_coherent_stimulation(&mut tick_input);

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
            let weights = inner_snapshot
                .synapse_states
                .iter()
                .map(|syn_state| syn_state.weight)
                .collect();

            Some(StateSnapshot {
                membrane_voltages,
                pre_syn_nids,
                post_syn_nids,
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
            self.add_non_coherent_stimulation(&mut tick_input);
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

    fn get_t(&self) -> usize {
        self.inner.get_tick_period()
    }
}

impl Instance {
    fn add_non_coherent_stimulation(&mut self, tick_input: &mut TickInput) {
        let num_spikes = (self.non_coherent_stimulation_rate
            * self.non_coherent_stimulation_nids.len() as f64) as usize;

        tick_input.force_spiking_nids.extend(
            self.non_coherent_stimulation_nids
                .choose_multiple(&mut self.rng, num_spikes)
                .copied(),
        )
    }
}

#[pyfunction]
fn create_from_yaml(yaml_str: &str) -> PyResult<Instance> {
    let params =
        serde_yaml::from_str(yaml_str).map_err(|error| PyValueError::new_err(error.to_string()))?;

    let instance = instance::create_instance(params)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let non_coherent_stimulation_nids =
        (0..(instance.get_num_neurons() - instance.get_num_out_channels())).collect::<Vec<_>>();

    Ok(Instance {
        inner: instance,
        reward_rate: 0.0,
        non_coherent_stimulation_rate: 0.0,
        non_coherent_stimulation_nids,
        rng: StdRng::seed_from_u64(0),
    })
}

#[pymodule]
fn morphyne(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_from_yaml, m)?)?;
    Ok(())
}
