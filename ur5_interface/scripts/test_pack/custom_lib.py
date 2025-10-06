from ur5_interface.msg import HumanJoint
import fcl
import numpy as np


class CollisionMonitor:
    def __init__(self):
        human_msg = HumanJoint()
        human_msg.exist = False
        self.update_static_env(human_msg)
        self.robot_param = (0.089159, 0.13585, -0.1197, 0.425, 0.39225, 0.10915, 0.093, 0.09465, 0.0823 + 0.005 + 0.26-0.02)

    def create_object(self, type: str, size: list, T=np.array([0, 0, 0], dtype=float), R=np.identity(3, dtype=float)):
        if type == "box":
            obj_type = fcl.Box(size[0], size[1], size[2])
        if type == "robot":
            obj_type = fcl.Sphere(0.025)
        if type == "human":
            obj_type = fcl.Sphere(0.0545)
        obj_transform = fcl.Transform(R, T)
        collision_obj = fcl.CollisionObject(obj_type, obj_transform)
        return collision_obj

    def update_static_env(self, human_msg: HumanJoint):
        self.box_2 = self.create_object("box", [0.28, 0.01, 0.22], T=np.array([0.45, -0.05 - 0.41 / 2, 0.11]))
        self.box_3 = self.create_object("box", [0.28, 0.01, 0.22], T=np.array([0.45, -0.05 + 0.41 / 2, 0.11]))
        self.box_4 = self.create_object("box", [0.01, 0.41, 0.22], T=np.array([0.45 + 0.28 / 2, -0.05, 0.11]))
        self.box_5 = self.create_object("box", [0.01, 0.41, 0.22], T=np.array([0.45 - 0.28 / 2, -0.05, 0.11]))

        self.static_env = [self.box_2, self.box_3, self.box_4, self.box_5]

        if human_msg.exist:
            self.human = self.create_object(
                "human",
                [0.6, 0.2, 0.3],
                T=np.array(
                    [
                        human_msg.position.x,
                        human_msg.position.y,
                        human_msg.position.z,
                    ]
                ),
            )
            self.static_env.append(self.human)

    def update_robot_model(self, pos_curr: list):

        th1, th2, th3, th4, th5, th6 = pos_curr
        d1, SO, EO, a2, a3, d4, d45, d5, d6 = self.robot_param

        PJ_2 = np.array(
            [
                -SO * np.sin(th1),
                SO * np.cos(th1),
                d1,
            ]
        )

        PJ_3 = np.array(
            [
                -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2),
                EO * np.cos(th1) + SO * np.cos(th1) + a2 * np.sin(th1) * np.cos(th2),
                -a2 * np.sin(th2) + d1,
            ]
        )
        PJ_4 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * (-np.sin(th2) * np.sin(th3) * np.cos(th1) + np.cos(th1) * np.cos(th2) * np.cos(th3)),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * (-np.sin(th1) * np.sin(th2) * np.sin(th3) + np.sin(th1) * np.cos(th2) * np.cos(th3)),
                -a2 * np.sin(th2) + a3 * (-np.sin(th2) * np.cos(th3) - np.sin(th3) * np.cos(th2)) + d1,
            ]
        )
        PJ_5 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * (-np.sin(th2) * np.sin(th3) * np.cos(th1) + np.cos(th1) * np.cos(th2) * np.cos(th3))
                - d45 * np.sin(th1),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * (-np.sin(th1) * np.sin(th2) * np.sin(th3) + np.sin(th1) * np.cos(th2) * np.cos(th3))
                + d45 * np.cos(th1),
                -a2 * np.sin(th2) + a3 * (-np.sin(th2) * np.cos(th3) - np.sin(th3) * np.cos(th2)) + d1,
            ]
        )
        PJ_6 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * np.cos(th1) * np.cos(th2 + th3)
                - d45 * np.sin(th1)
                - d5 * np.sin(th2 + th3 + th4) * np.cos(th1),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * np.sin(th1) * np.cos(th2 + th3)
                + d45 * np.cos(th1)
                - d5 * np.sin(th1) * np.sin(th2 + th3 + th4),
                -a2 * np.sin(th2) - a3 * np.sin(th2 + th3) + d1 - d5 * np.cos(th2 + th3 + th4),
            ]
        )
        PJ_7 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * np.cos(th1) * np.cos(th2 + th3)
                - d45 * np.sin(th1)
                - d5 * np.sin(th2 + th3 + th4) * np.cos(th1)
                - d6 * (np.sin(th1) * np.cos(th5) - np.sin(th5) * np.cos(th1) * np.cos(th2 + th3 + th4)),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * np.sin(th1) * np.cos(th2 + th3)
                + d45 * np.cos(th1)
                - d5 * np.sin(th1) * np.sin(th2 + th3 + th4)
                + d6 * (np.sin(th1) * np.sin(th5) * np.cos(th2 + th3 + th4) + np.cos(th1) * np.cos(th5)),
                -a2 * np.sin(th2) - a3 * np.sin(th2 + th3) + d1 - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4),
            ]
        )
        self.J2 = self.create_object("robot", [], PJ_2)
        self.J3 = self.create_object("robot", [], PJ_3)
        self.J4 = self.create_object("robot", [], PJ_4)
        self.J5 = self.create_object("robot", [], PJ_5)
        self.J6 = self.create_object("robot", [], PJ_6)
        self.J7 = self.create_object("robot", [], PJ_7)
        self.robot_curr_state = [self.J2, self.J3, self.J4, self.J5, self.J6, self.J7]

    def update_robot_goal(self, pos_goal: list):

        th1, th2, th3, th4, th5, th6 = pos_goal
        d1, SO, EO, a2, a3, d4, d45, d5, d6 = self.robot_param

        PJ_2 = np.array(
            [
                -SO * np.sin(th1),
                SO * np.cos(th1),
                d1,
            ]
        )

        PJ_3 = np.array(
            [
                -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2),
                EO * np.cos(th1) + SO * np.cos(th1) + a2 * np.sin(th1) * np.cos(th2),
                -a2 * np.sin(th2) + d1,
            ]
        )
        PJ_4 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * (-np.sin(th2) * np.sin(th3) * np.cos(th1) + np.cos(th1) * np.cos(th2) * np.cos(th3)),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * (-np.sin(th1) * np.sin(th2) * np.sin(th3) + np.sin(th1) * np.cos(th2) * np.cos(th3)),
                -a2 * np.sin(th2) + a3 * (-np.sin(th2) * np.cos(th3) - np.sin(th3) * np.cos(th2)) + d1,
            ]
        )
        PJ_5 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * (-np.sin(th2) * np.sin(th3) * np.cos(th1) + np.cos(th1) * np.cos(th2) * np.cos(th3))
                - d45 * np.sin(th1),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * (-np.sin(th1) * np.sin(th2) * np.sin(th3) + np.sin(th1) * np.cos(th2) * np.cos(th3))
                + d45 * np.cos(th1),
                -a2 * np.sin(th2) + a3 * (-np.sin(th2) * np.cos(th3) - np.sin(th3) * np.cos(th2)) + d1,
            ]
        )
        PJ_6 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * np.cos(th1) * np.cos(th2 + th3)
                - d45 * np.sin(th1)
                - d5 * np.sin(th2 + th3 + th4) * np.cos(th1),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * np.sin(th1) * np.cos(th2 + th3)
                + d45 * np.cos(th1)
                - d5 * np.sin(th1) * np.sin(th2 + th3 + th4),
                -a2 * np.sin(th2) - a3 * np.sin(th2 + th3) + d1 - d5 * np.cos(th2 + th3 + th4),
            ]
        )
        PJ_7 = np.array(
            [
                -EO * np.sin(th1)
                - SO * np.sin(th1)
                + a2 * np.cos(th1) * np.cos(th2)
                + a3 * np.cos(th1) * np.cos(th2 + th3)
                - d45 * np.sin(th1)
                - d5 * np.sin(th2 + th3 + th4) * np.cos(th1)
                - d6 * (np.sin(th1) * np.cos(th5) - np.sin(th5) * np.cos(th1) * np.cos(th2 + th3 + th4)),
                EO * np.cos(th1)
                + SO * np.cos(th1)
                + a2 * np.sin(th1) * np.cos(th2)
                + a3 * np.sin(th1) * np.cos(th2 + th3)
                + d45 * np.cos(th1)
                - d5 * np.sin(th1) * np.sin(th2 + th3 + th4)
                + d6 * (np.sin(th1) * np.sin(th5) * np.cos(th2 + th3 + th4) + np.cos(th1) * np.cos(th5)),
                -a2 * np.sin(th2) - a3 * np.sin(th2 + th3) + d1 - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4),
            ]
        )
        self.J2_goal = self.create_object("robot", [], PJ_2)
        self.J3_goal = self.create_object("robot", [], PJ_3)
        self.J4_goal = self.create_object("robot", [], PJ_4)
        self.J5_goal = self.create_object("robot", [], PJ_5)
        self.J6_goal = self.create_object("robot", [], PJ_6)
        self.J7_goal = self.create_object("robot", [], PJ_7)

        self.robot_goal_state = [self.J2_goal, self.J3_goal, self.J4_goal, self.J5_goal, self.J6_goal, self.J7_goal]


    def compute_Jacob(self, pos_curr: list):
        th1, th2, th3, th4, th5, th6 = pos_curr
        d1, SO, EO, a2, a3, d4, d45, d5, d6 = self.robot_param
        J_2_T = np.array(
            [
                [-SO * np.cos(th1), -SO * np.sin(th1), 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_3_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2),
                    0,
                ],
                [-a2 * np.sin(th2) * np.cos(th1), -a2 * np.sin(th1) * np.cos(th2), -a2 * np.cos(th2)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_4_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2) - a3 * np.sin(th1) * np.cos(th2 + th3),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2) + a3 * np.cos(th1) * np.cos(th2 + th3),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3),
                ],
                [-a3 * np.sin(th2 + th3) * np.cos(th1), -a3 * np.sin(th1) * np.sin(th2 + th3), -a3 * np.cos(th2 + th3)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_5_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2) - a3 * np.sin(th1) * np.cos(th2 + th3) - d45 * np.cos(th1),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2) + a3 * np.cos(th1) * np.cos(th2 + th3) - d45 * np.sin(th1),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3),
                ],
                [-a3 * np.sin(th2 + th3) * np.cos(th1), -a3 * np.sin(th1) * np.sin(th2 + th3), -a3 * np.cos(th2 + th3)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_6_p_T = np.asarray(
            [
                [
                    -EO * np.cos(th1)
                    - SO * np.cos(th1)
                    - a2 * np.sin(th1) * np.cos(th2)
                    - a3 * np.sin(th1) * np.cos(th2 + th3)
                    - d45 * np.cos(th1)
                    + d5 * np.sin(th1) * np.sin(th2 + th3 + th4),
                    -EO * np.sin(th1)
                    - SO * np.sin(th1)
                    + a2 * np.cos(th1) * np.cos(th2)
                    + a3 * np.cos(th1) * np.cos(th2 + th3)
                    - d45 * np.sin(th1)
                    - d5 * np.sin(th2 + th3 + th4) * np.cos(th1),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4),
                ],
                [
                    (-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.cos(th1),
                    (-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.sin(th1),
                    -a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4),
                ],
                [-d5 * np.cos(th1) * np.cos(th2 + th3 + th4), -d5 * np.sin(th1) * np.cos(th2 + th3 + th4), d5 * np.sin(th2 + th3 + th4)],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        gain_3 = 2.0
        J_7_T = np.asarray(
            [
                [
                    -EO * np.cos(th1)
                    - SO * np.cos(th1)
                    - a2 * np.sin(th1) * np.cos(th2)
                    - a3 * np.sin(th1) * np.cos(th2 + th3)
                    - d45 * np.cos(th1)
                    + d5 * np.sin(th1) * np.sin(th2 + th3 + th4)
                    - d6 * (np.sin(th1) * np.sin(th5) * np.cos(th2 + th3 + th4) + np.cos(th1) * np.cos(th5)),
                    -EO * np.sin(th1)
                    - SO * np.sin(th1)
                    + a2 * np.cos(th1) * np.cos(th2)
                    + a3 * np.cos(th1) * np.cos(th2 + th3)
                    - d45 * np.sin(th1)
                    - d5 * np.sin(th2 + th3 + th4) * np.cos(th1)
                    - d6 * (np.sin(th1) * np.cos(th5) - np.sin(th5) * np.cos(th1) * np.cos(th2 + th3 + th4)),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4))
                    * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4))
                    * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4),
                ],
                [
                    ((-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.cos(th1)) * gain_3,
                    ((-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.sin(th1)) * gain_3,
                    (-a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4)) * gain_3,
                ],
                [
                    (-d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.cos(th1),
                    (-d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.sin(th1),
                    d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4),
                ],
                [
                    d6 * (np.sin(th1) * np.sin(th5) + np.cos(th1) * np.cos(th5) * np.cos(th2 + th3 + th4)),
                    d6 * (np.sin(th1) * np.cos(th5) * np.cos(th2 + th3 + th4) - np.sin(th5) * np.cos(th1)),
                    -d6 * np.sin(th2 + th3 + th4) * np.cos(th5),
                ],
                [0, 0, 0],
            ]
        )

        self.jacob_list = [J_2_T, J_3_T, J_4_T, J_5_T, J_6_p_T, J_7_T]

        return self.jacob_list
    
    def compute_Jacob_goal(self, pos_goal: list):
        th1, th2, th3, th4, th5, th6 = pos_goal
        d1, SO, EO, a2, a3, d4, d45, d5, d6 = self.robot_param
        J_2_T = np.array(
            [
                [-SO * np.cos(th1), -SO * np.sin(th1), 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_3_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2),
                    0,
                ],
                [-a2 * np.sin(th2) * np.cos(th1), -a2 * np.sin(th1) * np.cos(th2), -a2 * np.cos(th2)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_4_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2) - a3 * np.sin(th1) * np.cos(th2 + th3),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2) + a3 * np.cos(th1) * np.cos(th2 + th3),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3),
                ],
                [-a3 * np.sin(th2 + th3) * np.cos(th1), -a3 * np.sin(th1) * np.sin(th2 + th3), -a3 * np.cos(th2 + th3)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_5_T = np.asarray(
            [
                [
                    -EO * np.cos(th1) - SO * np.cos(th1) - a2 * np.sin(th1) * np.cos(th2) - a3 * np.sin(th1) * np.cos(th2 + th3) - d45 * np.cos(th1),
                    -EO * np.sin(th1) - SO * np.sin(th1) + a2 * np.cos(th1) * np.cos(th2) + a3 * np.cos(th1) * np.cos(th2 + th3) - d45 * np.sin(th1),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3),
                ],
                [-a3 * np.sin(th2 + th3) * np.cos(th1), -a3 * np.sin(th1) * np.sin(th2 + th3), -a3 * np.cos(th2 + th3)],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        J_6_p_T = np.asarray(
            [
                [
                    -EO * np.cos(th1)
                    - SO * np.cos(th1)
                    - a2 * np.sin(th1) * np.cos(th2)
                    - a3 * np.sin(th1) * np.cos(th2 + th3)
                    - d45 * np.cos(th1)
                    + d5 * np.sin(th1) * np.sin(th2 + th3 + th4),
                    -EO * np.sin(th1)
                    - SO * np.sin(th1)
                    + a2 * np.cos(th1) * np.cos(th2)
                    + a3 * np.cos(th1) * np.cos(th2 + th3)
                    - d45 * np.sin(th1)
                    - d5 * np.sin(th2 + th3 + th4) * np.cos(th1),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4),
                ],
                [
                    (-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.cos(th1),
                    (-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4)) * np.sin(th1),
                    -a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4),
                ],
                [-d5 * np.cos(th1) * np.cos(th2 + th3 + th4), -d5 * np.sin(th1) * np.cos(th2 + th3 + th4), d5 * np.sin(th2 + th3 + th4)],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        gain_3 = 2.0
        J_7_T = np.asarray(
            [
                [
                    -EO * np.cos(th1)
                    - SO * np.cos(th1)
                    - a2 * np.sin(th1) * np.cos(th2)
                    - a3 * np.sin(th1) * np.cos(th2 + th3)
                    - d45 * np.cos(th1)
                    + d5 * np.sin(th1) * np.sin(th2 + th3 + th4)
                    - d6 * (np.sin(th1) * np.sin(th5) * np.cos(th2 + th3 + th4) + np.cos(th1) * np.cos(th5)),
                    -EO * np.sin(th1)
                    - SO * np.sin(th1)
                    + a2 * np.cos(th1) * np.cos(th2)
                    + a3 * np.cos(th1) * np.cos(th2 + th3)
                    - d45 * np.sin(th1)
                    - d5 * np.sin(th2 + th3 + th4) * np.cos(th1)
                    - d6 * (np.sin(th1) * np.cos(th5) - np.sin(th5) * np.cos(th1) * np.cos(th2 + th3 + th4)),
                    0,
                ],
                [
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4))
                    * np.cos(th1),
                    (-a2 * np.sin(th2) - a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4))
                    * np.sin(th1),
                    -a2 * np.cos(th2) - a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4),
                ],
                [
                    ((-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.cos(th1)) * gain_3,
                    ((-a3 * np.sin(th2 + th3) - d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.sin(th1)) * gain_3,
                    (-a3 * np.cos(th2 + th3) + d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4)) * gain_3,
                ],
                [
                    (-d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.cos(th1),
                    (-d5 * np.cos(th2 + th3 + th4) - d6 * np.sin(th5) * np.sin(th2 + th3 + th4)) * np.sin(th1),
                    d5 * np.sin(th2 + th3 + th4) - d6 * np.sin(th5) * np.cos(th2 + th3 + th4),
                ],
                [
                    d6 * (np.sin(th1) * np.sin(th5) + np.cos(th1) * np.cos(th5) * np.cos(th2 + th3 + th4)),
                    d6 * (np.sin(th1) * np.cos(th5) * np.cos(th2 + th3 + th4) - np.sin(th5) * np.cos(th1)),
                    -d6 * np.sin(th2 + th3 + th4) * np.cos(th5),
                ],
                [0, 0, 0],
            ]
        )

        self.goal_jacob_list = [J_2_T, J_3_T, J_4_T, J_5_T, J_6_p_T, J_7_T]

        return self.goal_jacob_list

    def compute_distance(self, o1, o2):
        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()
        ret = fcl.distance(o1, o2, request, result)

        return (ret, result.nearest_points[1] - result.nearest_points[0])

    def compute_torques(self, list_forces: list):
        torques_rep = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(6):
            torques_rep += np.dot(self.jacob_list[i], list_forces[i])

        if all(element == 0.0 for element in torques_rep):
            return torques_rep

        return torques_rep
    
    def compute_torques_goal(self, list_forces: list):
        torques_rep = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(6):
            torques_rep += np.dot(self.goal_jacob_list[i], list_forces[i])

        if all(element == 0.0 for element in torques_rep):
            return torques_rep

        return torques_rep

    def update_planning_scene(self, pos_curr, human_msg):
        self.update_robot_model(pos_curr)
        self.update_static_env(human_msg)
