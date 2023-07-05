from abc import ABC
from typing import List
import random
from revolve2.core.physics.actor import Actor, Joint
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from RNN_m2 import *
from Trial.project_m1.Get_states import retrieve_extended_joint_info
from revolve2.serialization import SerializeError, StaticData, Serializable
from OptimizerCentral_rnn import SIMULATION_TIME




class CBrain(Brain):
    """
    Centralized brain that controls each limb
    """

    def __init__(self, model):
        self._model = model

    # Pass the actions to the actor
    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        actor, _ = body.to_actor()

        return CController(actor, self._model)

class CController(ActorController, ABC):
    """
        Centralized controller for the brain
    """
    _actor: Actor

    def __init__(self, actor: Actor, model):
        self._actor = actor
        self._model = model

    # Get the targets
    def get_dof_targets(self) -> List[float]:
        new_movement = []
        single_joint_info = []
        inp_mod_try = []
        # Get the joint info
        for joint in self._actor.joints:
            single_joint_info += retrieve_extended_joint_info(joint)
        #print("joint info as input: ", single_joint_info)
        input_model = single_joint_info
        # Add randomness to fix freezing issue
        for inp in input_model:
            rand = random.uniform(-0.1, 0.1)
            inp += rand
            inp_mod_try.append(inp)
        input_model_tensor = torch.Tensor(inp_mod_try)
        output = self._model(input_model_tensor)
        # Clamp the output tensor
        output = torch.clamp(output, min=-0.5, max=0.5)
        output = output.tolist()
        # Return the targeted degree of movement
        return output

    def step(self, dt: float) -> None:
        pass

    def deserialize(cls, data: StaticData) -> Serializable:
        pass

    def serialize(self) -> StaticData:
        pass




