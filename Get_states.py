from revolve2.core.physics.actor import Joint, RigidBody, Actor
from typing import List


def retrieve_joint_info(joint: Joint) -> List:

    return [joint.position.x, joint.position.y, joint.position.z,
            joint.orientation.x, joint.orientation.y, joint.orientation.z, joint.orientation.w]


def retrieve_joint_name(joint: Joint) -> str:

    return joint.name


def retrieve_body_info(body: RigidBody) -> List:
    center = body.center_of_mass()
    return [body.position.x, body.position.y, body.position.z,
            body.orientation.x, body.orientation.y, body.orientation.z, body.orientation.w,
            center.x, center.y, center.z]


def retrieve_extended_joint_info(joint: Joint) -> List:
    info = retrieve_body_info(joint.body2)
    info.extend(retrieve_joint_info(joint))

    return info

def retrieve_extended_info_from_actor(actor: Actor):
    info = []
    for joint in actor.joints:
        info.append(retrieve_extended_joint_info(joint))
    return info

def retrieve_info_from_actor(actor: Actor):
    info_act = []
    for joint in actor.joints:
        info_act.append(retrieve_joint_info(joint))
    return info_act

def retrieve_joint_name_from_actor(actor: Actor):
    info_joint_name = []
    for joint in actor.joints:
        info_joint_name.append(retrieve_joint_name(joint))
    return info_joint_name

