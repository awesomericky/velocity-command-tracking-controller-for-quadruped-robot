import pdb

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys
    sys.path.insert(0, '/home/awesomericky/raisim/raisimLib/thirdParty/ompl-1.5.2/py-bindings')
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

import numpy as np

class Analytic_planner:
    """
    Generate instacle of corresponding class when new environment is generated
    """
    def __init__(self, env, map_size, max_planning_time, min_n_states=50):
        self.env = env
        self.map_size = map_size
        self.max_planning_time = max_planning_time  # seconds
        self.min_n_states = min_n_states
        self.path_coordinates = None

        # disable ompl log except error
        ou.setLogLevel(ou.LOG_ERROR)  # LOG_ERROR / LOG_WARN / LOG_INFO / LOG_DEBUG

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(- self.map_size / 2)
        bounds.setHigh(self.map_size / 2)
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        resolution_size = 0.05  # [m]
        si.setStateValidityCheckingResolution(resolution_size / si.getMaximumExtent())

        self.start = ob.State(space)
        self.goal = ob.State(space)

        self.planner = og.BITstar(si)  # create a planner for the defined space
        # self.planner = og.RRTstar(si)  # create a planner for the defined space
        self.pdef = ob.ProblemDefinition(si)  # create a problem instance

    """
    Value function : 
    로봇은 RRT 경로 위에 있어야함
    (1) 예측한 경로 길이 구한 후 구한 길이와 동일한 RRT 상의 경로 획득
    (2) nDTW(예측한 경로, 잘린 RRT 경로) + (RRT 경로 길이 / max 경로 길이) 를 (latent state, goal_pos)로 라벨링
    * max 경로 길이 = 12
    
    Path tracking
    (1) Threshold distance 내에서 N개의 goal 선택 (waypoint와 같이)
    """

    def isStateValid(self, state):
        """
        Set sphere and check collision

        :return:
        """
        if self.env is None:
            return True
        else:
            safe = not self.env.analytic_planner_collision_check(state[0], state[1])
            return safe
            # return not self.env.analytic_planner_collision_check(state[0], state[1])

    def plan(self, start, goal):
        """

        :param start: (2,) (numpy)
        :param goal: (2,) (numpy)
        :return:
        """
        self.start[0] = start[0]
        self.start[1] = start[1]
        self.goal[0] = goal[0]
        self.goal[1] = goal[1]

        self.pdef.setStartAndGoalStates(self.start, self.goal)
        self.planner.setProblemDefinition(self.pdef)  # set the problem we are trying to solve for the planner
        self.planner.setup()  # perform setup steps for the planner

        solved = self.planner.solve(self.max_planning_time)

        if solved:
            path = self.pdef.getSolutionPath()
            path.interpolate(self.min_n_states)
            n_states = path.getStateCount()
            path_states = path.getStates()

            self.path_coordinates = np.zeros((n_states, 2))
            for i in range(n_states):
                self.path_coordinates[i][0] = path_states[i][0]
                self.path_coordinates[i][1] = path_states[i][1]
            self.path_coordinates = self.path_coordinates.astype(np.float32)
            return self.path_coordinates.copy()
        else:
            print("No solution found")
            return None

    def visualize_path(self):
        self.env.visualize_analytic_planner_path(self.path_coordinates)

if __name__ == "__main__":
    planner = Analytic_planner(env=None, map_size=40., max_planning_time=10.)
    start = [0., 0.]
    goal = [18., 18.]
    path = planner.plan(start, goal)
    print(path)
