N = 10000

num_I_neurons = int(N * 0.2)
num_E_neurons = N - num_I_neurons

conn_EX = {
    "from_layer_id": 0,
    "conduction_delay_position_distance_scale_factor": 0,
    "projection_params": {
        "synapse_params": {
            "weight_scale_factor": 1,
            "max_weight": 0.5
        },
        "stp_params": "NoStp",
        "long_term_stdp_params": {
            "factor_pre_before_post": 0.1,
            "tau_pre_before_post": 20,
            "factor_pre_after_post": -0.12,
            "tau_pre_after_post": 20,
        }
    },
    "connect_density": 0.01,
    "connect_width": 2,
    "conduction_delay_max_random_part": 10,
    "conduction_delay_add_on": 0,
    "allow_self_innervation": False,
    "initial_syn_weight": {
        "Constant": 0.6
    }
}

conn_IX = {
    "from_layer_id": 1,
    "conduction_delay_position_distance_scale_factor": 0,
    "projection_params": {
        "synapse_params": {
            "weight_scale_factor": -1,
            "max_weight": 10.0
        },
        "stp_params": "NoStp"
    },
    "connect_density": 0.1,
    "connect_width": 2,
    "conduction_delay_max_random_part": 0,
    "conduction_delay_add_on": 0,
    "allow_self_innervation": False,
    "initial_syn_weight": {
        "Constant": 2.0
    }
}

plasticity_modulation_params = {
    "tau_eligibility_trace": 2,
    "eligibility_trace_delay": 0,
    "t_cutoff_eligibility_trace": 2,
    "dopamine_flush_period": 1,
    "dopamine_conflation_period": 1,
    "dopamine_modulation_factor": 1.0
}

# plasticity_modulation_params = None

example_params = {
    "layers": [
        {
            "num_neurons": num_E_neurons,
            "neuron_params": {
                "tau_membrane": 20,
                "refractory_period": 10,
                "reset_voltage": 0,
                "t_cutoff_coincidence": 25,
                "adaptation_threshold": 1,
                "tau_threshold": 50,
                "voltage_floor": -10
            },
            "plasticity_modulation_params": plasticity_modulation_params,
        },
        {
            "num_neurons": num_I_neurons,
            "neuron_params": {
                "tau_membrane": 5,
                "refractory_period": 1,
                "reset_voltage": 0,
                "t_cutoff_coincidence": 1,
                "adaptation_threshold": 1,
                "tau_threshold": 50,
                "voltage_floor": 0
            },
            "plasticity_modulation_params": plasticity_modulation_params,
        },
    ],
    "layer_connections": [
        conn_EX | {"to_layer_id": 0},
        conn_EX | {"to_layer_id": 1},
        conn_IX | {"to_layer_id": 0},
        conn_IX | {"to_layer_id": 1},

    ],
    "technical_params": {
        "num_threads": 10,
        "pin_threads": False,
        "seed_override": 0,
    }
}


def get_example_params() -> dict:
    return example_params
