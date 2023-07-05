import math
import numpy as np
from random import Random
from revolve2.core.physics.environment_actor_controller import EnvironmentActorController
from revolve2.standard_resources import terrains
from pyrr import Quaternion, Vector3
from revolve2.core.modular_robot import ActiveHinge, Body, Brick, ModularRobot
import pandas as pd
from revolve2.core.physics.running import Batch, Environment, PosedActor, ActorState, Runner
from revolve2.runners.mujoco import LocalRunner
from Trial.project_m1.project_m2.C_Brain_Rnn import CBrain
from DC_Brain_rnn import DcBrain
from Trial.project_m1.project_m2.RNN_m2 import RNN
import torch
FITNESS_LIST = []
SIMULATION_TIME: 100

# Initiate Simulator
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
        """
        batch = Batch(
            simulation_time=100,
            sampling_frequency=15,
            control_frequency=15,
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


# Calculate the distance traveled by the agent
def _calculate_distance(begin_state: ActorState, end_state: ActorState) -> float:
    # distance traveled on the xy plane
    return math.sqrt(
        (begin_state.position[0] - end_state.position[0]) ** 2
        + ((begin_state.position[1] - end_state.position[1]) ** 2)
    )


async def main() -> None:
    """Run the simulation."""
    # initiate the body
    rng = Random()
    rng.seed(5)

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
    sum = 0
    # initiate the brain
    for i in range(1):
        print("sim number: ", i)
        # load model that you want to use
        rnn = torch.load('model_decentral_2.pth')
        #rnn = RNN(102, 200, 6)
        # Initiate if you want centralized or decentralized controller
        brain = DcBrain(rnn)
        #brain = DcBrain(rnn)
        robot = ModularRobot(body, brain)
        sim = Simulator()
        await sim.simulate(robot)
    # Uncomment if you want to store the fitness data
    #for num in FITNESS_LIST:
        #if num > 7:
        #sum += num
        #else:
            #sum += 7
    #df = pd.DataFrame()
    #df["fitness"] = FITNESS_LIST
    #avg = sum / len(FITNESS_LIST)
    #print(avg)
    #df.to_excel('performances_decentral_3.xlsx', index=False)

# Run main
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())







