from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt
import copy
import torch
from revolve2.core.physics.actor import Actor, Joint
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Body, Brain
from Trial.project_m1.RNN import Rnn, Linear
from Trial.project_m1.Get_states import retrieve_extended_joint_info, retrieve_joint_name_from_actor
from revolve2.serialization import SerializeError, StaticData, Serializable
import math
from revolve2.core.physics.running import (
    ActorControl,
    Batch,
    Environment,
    EnvironmentController,
    PosedActor,
    ActorState,
)
import random


"""
# Manualy update weights
for param in mask_model.parameters():
    param.data = nn.parameter.Parameter(torch.ones_like(param))"""


class DcBrain(Brain):
    """
    Decentralized brain that controls each limb
    """
    def __init__(self, model):
        self._model = model

    # Pass the actions to the actor
    def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
        actor, _ = body.to_actor()

        return DcController(actor, self._model)


class DcController(ActorController, ABC):
    """
    Decentralized controller for the brain
    """
    _actor: Actor

    def __init__(self, actor: Actor, model):
        self._actor = actor
        self._model = model

    # Get the targets
    def get_dof_targets(self) -> List[float]:
        new_movement = []
        single_joint_info = []
        fin_movement = []
        # Get the joint info
        for joint in self._actor.joints:
            single_joint_info += retrieve_extended_joint_info(joint)

        # print("joint info as input: ", single_joint_info)
        # Map the neighbors joint info
        joint_3 = [single_joint_info[0:17]] # origin_back_active_hinge
        joint_4 = [single_joint_info[17:34]] # origin_back_attachment_front_active_hinge
        joint_5 = [single_joint_info[34:51]] # origin_back_attachment_front_attachment_left_active_hinge
        joint_6 = [single_joint_info[51:68]] # origin_back_attachment_front_attachment_right_active_hinge
        joint_1 = [single_joint_info[68:85]] # origin_left_active_hinge
        joint_2 = [single_joint_info[85:102]] # origin_right_active_hinge
        joint_list = [joint_3, joint_4, joint_5, joint_6, joint_1, joint_2]

        #print("inputs for each joint: ")
        # Run for every joint
        for i in joint_list:
            input_model = []
            inp_mod_2 = []
            inp_try = []
            if i == joint_3: # 3, 4, 1, 2
                input_model += joint_3
                input_model += joint_4
                input_model += joint_2
                input_model += joint_1
            elif i == joint_4: # 3 4 5 6
                input_model += joint_3
                input_model += joint_4
                input_model += joint_5
                input_model += joint_6
            elif i == joint_5: # 5 6 4 3
                input_model += joint_5
                input_model += joint_6
                input_model += joint_4
                input_model += joint_3
            elif i == joint_6: # 6 5 4 3
                input_model += joint_6
                input_model += joint_5
                input_model += joint_4
                input_model += joint_3
            elif i == joint_1: # 1 2 3 4
                input_model += joint_1
                input_model += joint_2
                input_model += joint_3
                input_model += joint_4
            elif i == joint_2: # 2 1 3 4
                input_model += joint_2
                input_model += joint_1
                input_model += joint_3
                input_model += joint_4
            for it in input_model:
                inp_mod_2 += it

            # Add slight randomness to inputs
            for inp in inp_mod_2:
                rand = random.uniform(-0.1, 0.1)
                inp += rand
                inp_try.append(inp)

            #print("model_input: ", inp_mod_2)
            #print("model_input_changed: ", inp_try)
            input_model_tensor = torch.Tensor(inp_try)
            output = self._model(input_model_tensor)
            # Dof ranges
            # Clamp the output
            output = torch.clamp(output, min=-0.5, max=0.5)
            no = output.tolist()
            new_movement += no
            new_movement_2 = torch.Tensor(new_movement)
        #print(new_movement_2)
        return new_movement_2

    def step(self, dt: float) -> None:
        pass

    def deserialize(cls, data: StaticData) -> Serializable:
        pass

    def serialize(self) -> StaticData:
        pass



