# Args standard
# to modify (support warm start, dynamic add args item)

# args keywords
MUST_KEYS = ["learning_rate", "..."]

# args data type
DATA_TYPE = [float, "..."]

# args data domain
# Continuous: tuple(lower bound, upper bound)
# Discrete: list[val1, val2, val3]
DATA_DOMAIN = [(0, 1), [1, 2, 3]]

# args description
DATA_HELP = ["", ""]

# Dict---Key:arg keywords, Val:tuple(data type, data domain)
# Val[0]: datatype, Val[1]: data domain
ARGS_STANDARD = dict(zip(MUST_KEYS, zip(DATA_TYPE, DATA_DOMAIN, DATA_HELP)))

DEFAULT_ARGS = {
                "use_gpu": True,
                "gpu_ids": [0],

                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "nesterov": True,

                "step_size": 1,
                "gamma": 0.5 ** (1 / 100),

                "loss_func": 'cross_entropy',

                "batch_limit": 99999,
                "coff": 3,
                "mu": 0.1
}
