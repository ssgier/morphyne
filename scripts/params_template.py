params_yaml = """
layers:
- num_neurons: 800
  neuron_params:
    tau_membrane: 10.0
    refractory_period: 10
    reset_voltage: 0.0
    t_cutoff_coincidence: 20
    adaptation_threshold: 1.0
    tau_threshold: 50.0
    voltage_floor: 0.0
  plasticity_modulation_params:
    tau_eligibility_trace: 1000.0
    eligibility_trace_delay: 20
    dopamine_modulation_factor: 1.5
    t_cutoff_eligibility_trace: 1500
    dopamine_flush_period: 250
    dopamine_conflation_period: 250
- num_neurons: 200
  neuron_params:
    tau_membrane: 4.0
    refractory_period: 5
    reset_voltage: 0.0
    t_cutoff_coincidence: 20
    adaptation_threshold: 1.0
    tau_threshold: 50.0
    voltage_floor: 0.0
  plasticity_modulation_params: null
layer_connections:
- from_layer_id: 0
  to_layer_id: 0
  projection_params:
    synapse_params:
      max_weight: 0.5
      weight_scale_factor: 1.0
    stp_params: !Depression
      tau: 800.0
      p0: 0.9
      factor: 0.1
    long_term_stdp_params:
      factor_pre_before_post: 0.1
      tau_pre_before_post: 20.0
      factor_pre_after_post: -0.12
      tau_pre_after_post: 20.0
    short_term_stdp_params:
      stdp_params:
        factor_pre_before_post: 0.01
        tau_pre_before_post: 20.0
        factor_pre_after_post: 0.012
        tau_pre_after_post: 20.0
      tau: 500.0
  connect_density: 0.1
  connect_width: 2.0
  initial_syn_weight: !Randomized 0.5
  conduction_delay_max_random_part: 20
  conduction_delay_position_distance_scale_factor: 0.0
  conduction_delay_add_on: 0
  allow_self_innervation: true
- from_layer_id: 0
  to_layer_id: 1
  projection_params:
    synapse_params:
      max_weight: 0.5
      weight_scale_factor: 2.0
    stp_params: !Depression
      tau: 800.0
      p0: 0.9
      factor: 0.1
    long_term_stdp_params:
      factor_pre_before_post: 0.1
      tau_pre_before_post: 20.0
      factor_pre_after_post: -0.12
      tau_pre_after_post: 20.0
    short_term_stdp_params:
      stdp_params:
        factor_pre_before_post: 0.01
        tau_pre_before_post: 20.0
        factor_pre_after_post: 0.012
        tau_pre_after_post: 20.0
      tau: 500.0
  connect_density: 0.25
  connect_width: 2.0
  initial_syn_weight: !Randomized 0.5
  conduction_delay_max_random_part: 20
  conduction_delay_position_distance_scale_factor: 0.0
  conduction_delay_add_on: 0
  allow_self_innervation: true
- from_layer_id: 1
  to_layer_id: 0
  projection_params:
    synapse_params:
      max_weight: 0.5
      weight_scale_factor: -1.0
    stp_params: NoStp
    long_term_stdp_params: null
    short_term_stdp_params: null
  connect_density: 0.25
  connect_width: 2.0
  initial_syn_weight: !Constant 0.85
  conduction_delay_max_random_part: 0
  conduction_delay_position_distance_scale_factor: 0.0
  conduction_delay_add_on: 0
  allow_self_innervation: true
- from_layer_id: 1
  to_layer_id: 1
  projection_params:
    synapse_params:
      max_weight: 0.5
      weight_scale_factor: -1.0
    stp_params: NoStp
    long_term_stdp_params: null
    short_term_stdp_params: null
  connect_density: 0.25
  connect_width: 2.0
  initial_syn_weight: !Constant 0.85
  conduction_delay_max_random_part: 0
  conduction_delay_position_distance_scale_factor: 0.0
  conduction_delay_add_on: 0
  allow_self_innervation: true
technical_params:
  num_threads: 1
  pin_threads: false
  batched_ring_buffer_size: 21
"""
