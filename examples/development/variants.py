from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'DClaw3': 250,
    'ImageDClaw3': 100,
    'HardwareDClaw3': 100,
    'Pendulum': 200,
    'Pusher2d': 100,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(3e2),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'DClaw3': int(200),
    'ImageDClaw3': int(300),
    'HardwareDClaw3': int(100),
    'Pendulum': 10,
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': tune.sample_from(lambda spec: (
                    10 * spec.get('config', spec)
                    ['sampler_params']
                    ['kwargs']
                    ['max_path_length']
                )),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}


class NegativeLogLossFn(object):
    def __init__(self, eps, offset=0.0):
        self._eps = eps
        self._offset = offset

    def __call__(self, object_target_distance):
        return - np.log(object_target_distance + self._eps) + self._offset

    def __str__(self):
        return (
            f'NegativeLogLossFn(eps={self._eps:e},offset={self._offset:.3f})')

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._eps == other._eps and self._offset == other._offset

        return super(NegativeLogLossFn, self).__eq__(other)


class LinearLossFn(object):
    def __call__(self, object_target_distance):
        return -object_target_distance

    def __str__(self):
        return str(f'LinearLossFn()')

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._eps == other._eps

        return super(LinearLossFn, self).__eq__(other)


ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v3': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'DClaw3': {
        'ScrewV2-v0': {
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        },
        'ImageScrewV2-v0': {
            'image_shape': (32, 32, 3),
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'HardwareDClaw3': {
        'ScrewV2-v0': {
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-1.98, -1.98 + 2 * np.pi),
        },
        'ImageScrewV2-v0': {
            'image_shape': (32, 32, 3),
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-1.98, -1.98 + 2 * np.pi),
        },
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    }
}

NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(5e5),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': 1000,
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(lambda spec: (
                25000 // (spec.get('config', spec)
                          ['algorithm_params']
                          ['kwargs']
                          ['epoch_length'])
            )),
            # NUM_EPOCHS_PER_DOMAIN.get(
            #     domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS
        },
    }

    return variant_spec


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    image_shape = (
        variant_spec
        ['environment_params']
        ['training']
        ['kwargs']
        ['image_shape'])

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_type = "conv"
        if preprocessor_type == "conv":
            preprocessor_params = tune.grid_search([
                {
                    'type': 'ConvnetPreprocessor',
                    'kwargs': {
                        'image_shape': image_shape,
                        'output_size': None,
                        'conv_filters': (base_size, ) * num_layers,
                        'conv_kernel_sizes': (conv_kernel_size, ) * num_layers,
                        'conv_strides': (conv_strides, ) * num_layers,
                        'normalization_type': normalization_type,
                        'downsampling_type': downsampling_type,
                        'use_global_average_pool': use_global_average_pool,
                    },
                }
                for base_size in (64, )
                for conv_kernel_size in (3, )
                for conv_strides in (2, )
                for normalization_type in (None, )
                for num_layers in (3, )
                for use_global_average_pool in (False, )
                for downsampling_type in ('conv', )
                if (image_shape[0] / (conv_strides ** num_layers)) >= 1
            ])
        elif preprocessor_type == "vae":
            num_layers = 4
            preprocessor_params = {
                'type': 'VAEPreprocessor',
                'kwargs': {
                    'image_shape': (
                        variant_spec
                        ['environment_params']
                        ['training']
                        ['kwargs']
                        ['image_shape']),
                    'conv_filters': (64, ) * num_layers,
                    'conv_kernel_sizes': (3, ) * num_layers,
                    'conv_strides': (2, ) * num_layers,
                    'normalization_type': None,
                    'downsampling_type': 'conv',
                    'output_size': 16,
                    'beta': tune.grid_search([1.0, 3.0, 10.0, 30.0, 100.0]),
                    'loss_weight': tune.grid_search([1e-3, 1e-2, 1e-1, 0.0]),
                },
            }
        else:
            raise NotImplementedError(preprocessor_type)

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['preprocessor_params']
            )))

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    if ('image' in task.lower()
        or 'blind' in task.lower()
        or 'image' in domain.lower()):
        variant_spec = get_variant_spec_image(
            universe, domain, task, args.policy, args.algorithm)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
