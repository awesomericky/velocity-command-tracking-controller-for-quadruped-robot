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

class Analytic_planner:
    """
    Generate instacle of corresponding class when new environment is generated
    """
    def __init__(self, heightmap, map_size, max_planning_time):
        self.heightmap = heightmap
        self.map_size = map_size
        self.max_planning_time = max_planning_time  # seconds

        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(- self.map_size / 2)
        bounds.setHigh(self.map_size / 2)
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))

        self.start = ob.State(space)
        self.goal = ob.State(space)

        self.planner = og.BITstar(si)  # create a planner for the defined space
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

    def isStateValid(state):
        """
        Set sphere and check collision

        :return:
        """
        return True

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
            return path
        else:
            print("No solution found")
            return None

