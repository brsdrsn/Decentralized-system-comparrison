import torch
from revolve2.runners.mujoco import LocalRunner
from revolve2.core.physics.running import Batch, Environment, PosedActor, ActorState, Runner

import math
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.standard_resources import terrains
from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
import random
from collections import OrderedDict
from RNN_m2 import *

# Initiate parameters
POPULATION_SIZE = 40
OFFSPRING_SIZE = 40
NUM_GENERATIONS = 69
NEXT_GENERATION = []
WEIGHT_LIST = []
FITNESS_LIST = []
SIMULATION_TIME = 100

# Get the distance
def _calculate_distance(begin_state: ActorState, end_state: ActorState) -> float:
    # distance traveled on the xy plane
    return math.sqrt(
        (begin_state.position[0] - end_state.position[0]) ** 2
        + ((begin_state.position[1] - end_state.position[1]) ** 2))

"""
def get_fittest_individual_indices(fitness_list):
    best_pop_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i])[-(int(OFFSPRING_SIZE / 2)):]
    return best_pop_indices


def create_new_pop(fittest_indices, model_list):
    next_generation = []
    curr_fittest = []
    # get the current fittest
    for i in fittest_indices:
        next_fittest = model_list[i]
        curr_fittest.append(next_fittest)
    next_generation += curr_fittest
    # mutate the current fittest
    #curr_to_use = curr_fittest
    #mutated_pop = mutate(curr_to_use)
    #next_generation += mutated_pop
    #mutated_pop2 = mutate(curr_to_use)
    # crossover the current fittest
    #crossover_ind = crossover(curr_fittest)
    #cross_pop.append(crossover_ind)
    #next_generation += mutated_pop2
    #next_generation += cross_pop
    return next_generation
"""
"""
def mutate(curr_this):
    mutated_models = []
    curr_fittest = curr_this
    for ind in curr_fittest:
        # 1st layer
        list_layer_1 = list(ind.parameters())[0].tolist()
        new_list_layer_1 = []
        for w in list_layer_1:
            new_list_layer_1_list = []
            for x in w:
                rand = random.uniform(-0.5, 0.5)
                x = x + rand
                new_list_layer_1_list.append(x)
            new_list_layer_1.append(new_list_layer_1_list)
        list_layer_1_tensor = torch.Tensor(new_list_layer_1)

        # 2st layer
        list_layer_2 = list(ind.parameters())[1].tolist()
        new_list_layer_2 = []
        for w in list_layer_2:
            new_list_layer_2_list = []
            for x in w:
                rand = random.uniform(-0.5, 0.5)
                x = x + rand
                new_list_layer_2_list.append(x)
            new_list_layer_2.append(new_list_layer_2_list)
        list_layer_2_tensor = torch.Tensor(new_list_layer_2)

        # 3st layer
        list_layer_3 = list(ind.parameters())[2].tolist()
        new_list_layer_3 = []
        for x in list_layer_3:
            rand = random.uniform(-0.5, 0.5)
            x = x + rand
            new_list_layer_3.append(x)
        list_layer_3_tensor = torch.Tensor(new_list_layer_3)

        # 4st layer
        list_layer_4 = list(ind.parameters())[3].tolist()
        new_list_layer_4 = []
        for x in list_layer_4:
            rand = random.uniform(-0.5, 0.5)
            x = x + rand
            new_list_layer_4.append(x)
        list_layer_4_tensor = torch.Tensor(new_list_layer_4)

        # 5rd layer
        list_layer_5 = list(ind.parameters())[4].tolist()
        new_list_layer_5 = []
        for w in list_layer_5:
            new_list_layer_5_list = []
            for x in w:
                rand = random.uniform(-0.5, 0.5)
                x = x + rand
                new_list_layer_5_list.append(x)
            new_list_layer_5.append(new_list_layer_5_list)
        list_layer_5_tensor = torch.Tensor(new_list_layer_5)

        # 6st layer
        list_layer_6 = list(ind.parameters())[5].tolist()
        new_list_layer_6 = []
        for x in list_layer_6:
            rand = random.uniform(-0.5, 0.5)
            x = x + rand
            new_list_layer_6.append(x)
        list_layer_6_tensor = torch.Tensor(new_list_layer_6)

        new_state_dict = OrderedDict({'rnn.weight_ih_l0': list_layer_1_tensor,
                                      'rnn.weight_hh_l0': list_layer_2_tensor,
                                      'rnn.bias_ih_l0': list_layer_3_tensor,
                                      'rnn.bias_hh_l0': list_layer_4_tensor,
                                      'fc.weight': list_layer_5_tensor,
                                      'fc.bias ': list_layer_6_tensor
                                      })

        ind.load_state_dict(new_state_dict, strict=False)
        mutated_models.append(ind)
    return mutated_models

"""
"""____________________________OPTIMIZER____________________________"""


def fitness_clear():
    FITNESS_LIST.clear()

# Initiate simulaiton
class OptSimulatorCentral:
    _runner: Runner

    def __init__(self, model_list):
        self.model_list = model_list
    """
    Simulator setup.

    Simulates using Mujoco.
    Defines a control function that steps the controller and applies the degrees of freedom the controller provides.
    """

    async def simulate(self, robot: ModularRobot) -> None:
        """
        Simulate a robot.

        :param robot: The robot to simulate.
        """
        batch = Batch(
            simulation_time=SIMULATION_TIME,
            sampling_frequency=60,
            control_frequency=5,
        )

        actor, controller = robot.make_actor_and_controller()
        bounding_box = actor.calc_aabb()
        env = Environment(EnvironmentActorController(controller))
        env.static_geometries.extend(terrains.flat().static_geometry)
        env.actors.append(
            PosedActor(
                actor,
                Vector3(
                    [
                        0.0,
                        0.0,
                        bounding_box.size.z / 2.0 - bounding_box.offset.z,
                    ]
                ),
                Quaternion(),
                [0.0 for _ in controller.get_dof_targets()],
            )
        )
        batch.environments.append(env)
        runner = LocalRunner(headless=True)
        batch_results = await LocalRunner.run_batch(self=runner, batch=batch, record_settings=None)
        for environment_result in batch_results.environment_results:
            FITNESS_LIST.append(_calculate_distance(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0], ))
        #await runner.run_batch(batch)
        #print("FITNESS_LIST: ", FITNESS_LIST)
        #print("LEN_FITNESS_LIST: ", len(FITNESS_LIST))
        #current_fittest_indices = get_fittest_individual_indices(FITNESS_LIST)
        #print("BEST_FITNESS_INDICES: ", current_fittest_indices)
        #NEXT_GENERATION.clear()
        #NEXT_GENERATION.append(create_new_pop(fittest_indices=current_fittest_indices, model_list=self.model_list))
        # print("BEST_POPULATION_MODELS", new_population_models)
        # print("BEST_POPULATION_MODELS_LENGTH", len(new_population_models))
        #NEXT_GENERATION.clear()
        #NEXT_GENERATION.append(new_population_models)


