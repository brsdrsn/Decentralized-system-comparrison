import math
import numpy as np
from random import Random
from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
import pandas as pd

import random
from collections import OrderedDict
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.core.physics.running import (
    ActorControl,
    Runner,
    Batch,
    Environment,
    EnvironmentController,
    PosedActor,
    ActorState,
)

from revolve2.runners.mujoco import LocalRunner

from revolve2.standard_resources import terrains

from Optimizer_rnn import NEXT_GENERATION, NUM_GENERATIONS, fitness_clear, FITNESS_LIST, POPULATION_SIZE, \
    OFFSPRING_SIZE, OptSimulatorCentral
from C_Brain_Rnn import CBrain
from RNN_m2 import RNN
import torch

# Initiates a simulator. Not really nescessary for the latest version since we initiate it in the Optimizer_rnn.py
class Simulator:
    _runner: Runner
    """
    Simulator setup.

    Simulates using Mujoco.
    Defines a control function that steps the controller and applies the degrees of freedom the controller provides.
   """

    async def simulate(self, robot: ModularRobot) -> None:
        """
        Simulate a robot.

        :param robot: The robot to simulate.
        :param control_frequency: Control frequency for the simulator.
        """
        batch = Batch(
            simulation_time=100,
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
        runner = LocalRunner(headless=False)
        batch_results = await LocalRunner.run_batch(self=runner, batch=batch, record_settings=None)
        for environment_result in batch_results.environment_results:
            FITNESS_LIST.append(_calculate_distance(
                environment_result.environment_states[0].actor_states[0],
                environment_result.environment_states[-1].actor_states[0], ))
        await runner.run_batch(batch)

# Calculates distance traveled by agent
def _calculate_distance(begin_state: ActorState, end_state: ActorState) -> float:
    # distance traveled on the xy plane
    return math.sqrt(
        (begin_state.position[0] - end_state.position[0]) ** 2
        + ((begin_state.position[1] - end_state.position[1]) ** 2)
    )


async def main() -> None:
    """Run the simulation."""
    rng = Random()
    rng.seed(5)
    # Creates agent
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    body.finalize()
    # Create the RNN
    def create_module(inp_size, hidden_size, out_size):
        return RNN(inp_size, hidden_size, out_size)

    # Create the RNN list
    def create_model_list():
        models_init = []
        for i in range(POPULATION_SIZE):
            m = create_module(102, 200, 6)
            models_init.append(m)
        return models_init

    # Get the position of the best individuals in the list
    def get_fittest_individual_indices(fitness_list):
        best_pop_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i])[-(int(OFFSPRING_SIZE / 4)):]
        return best_pop_indices

    # Get the best individuals
    def select_best_from_pop(fittest_indices, current_model_list):
        print("forming next generation")
        next_generation = []
        curr_fittest = []
        # get the current fittest
        for i in fittest_indices:
            next_fittest = current_model_list[i]
            curr_fittest.append(next_fittest)
        next_generation += curr_fittest
        return next_generation

    # Mutate the best individuals
    def mutate(curr_fittest):
        mutated_models = []

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
            new_ind = RNN(102, 200, 6)
            new_ind.load_state_dict(new_state_dict, strict=False)
            mutated_models.append(new_ind)
        return mutated_models

    model_list = create_model_list()
    best_fitness_list = []

    # Train the models
    for i in range(NUM_GENERATIONS):
        next_generation = []
        print("_______________GENERATION:", i + 1, "_______________")
        for i in range(len(model_list)):
            print("On module number:", i + 1)
            # initiate the model to use
            model_to_use = model_list[i]
            brain = CBrain(model_to_use)
            robot = ModularRobot(body, brain)
            sim = OptSimulatorCentral(model_list)
            await sim.simulate(robot)
        # Get the best fitness score
        best_fitness_score = max(FITNESS_LIST)
        print("best fitness of this generation:", best_fitness_score)
        current_best_fittest_indices = get_fittest_individual_indices(FITNESS_LIST)
        fittest_inds = select_best_from_pop(current_best_fittest_indices, model_list)
        # Append the fittest individuals to the next generation
        next_generation += fittest_inds
        next_generation += fittest_inds
        # Append the mutated individuals to the next generation
        mutated_inds = mutate(fittest_inds)
        next_generation += mutated_inds
        mutated_inds2 = mutate(fittest_inds)
        next_generation += mutated_inds2
        # Update the population
        model_list = next_generation
        best_fitness_list.append(best_fitness_score)
        # Get the best individual
        best_one_index = max(enumerate(FITNESS_LIST), key=lambda x: x[1])[0]
        fitness_clear()
    # Save the results(model and fitness list)
    print("the best fitness scores from every generation: ", best_fitness_list)
    df = pd.DataFrame()
    df["fitness"] = best_fitness_list
    df.to_excel('result_central_6.xlsx', index=False)
    best_model = model_list[best_one_index]
    torch.save(best_model, 'model_central_6.pth')

    # model_list = NEXT_GENERATION[0]
    # Brain of the robot
    # brain = BrainCpgNetworkNeighbourRandom(rng)


# Run the program
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())









