# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def pathMapToList(pathMap : dict, goal : tuple) -> list:
    """
    Converts a path map to a list of actions.
    """
    path = []
    current = goal
    while current != None:
        path.append(current[1])
        current = pathMap.get(current)

    return list(reversed(path[:-1]))


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    path = {}
    visited = []

    stack = util.Stack()
    stack.push((problem.getStartState(), None))

    # DFS
    while (not stack.isEmpty()):
        current = stack.pop()

        if problem.isGoalState(current[0]):
            return pathMapToList(path, current)

        if current[0] in visited:
            continue

        visited.append(current[0])

        for action in problem.expand(current[0]):
            # Ommit score
            nextNode = (action[0], action[1])  
            stack.push(nextNode)

            # Update path map
            path[nextNode] = current           
           
    # No path
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    path = {}
    visited = []
    
    queue = util.Queue()
    queue.push((problem.getStartState(), None))

    # BFS 
    while (not queue.isEmpty()):
        current = queue.pop()

        if problem.isGoalState(current[0]):
            return pathMapToList(path, current)

        if current[0] in visited:
            continue

        visited.append(current[0])

        for action in problem.expand(current[0]):
            # Ommit score
            nextNode = (action[0], action[1])  
            queue.push(nextNode)

            # Update path map
            path[nextNode] = current
    
    # No path
    return []
    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarEvaluation(problem, state, cost : int, heuristic) -> int:
    return cost + heuristic(state[0], problem)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    path = {}
    pathCost = {problem.getStartState(): 0}
    fCost = {problem.getStartState(): 0}
    visited = []

    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), None, 0), 0)

    # A* Search
    while(not queue.isEmpty()):
        current = queue.pop()

        # Updating accumulated cost of traversal
        currentCost = pathCost.get(current[0])

        if problem.isGoalState(current[0]):
            return pathMapToList(path, current)

        if current[0] in visited:
            continue

        visited.append(current[0])

        for action in problem.expand(current[0]):
            
            newPathCost = action[2] + currentCost
            nextState = action[0]
            
            # If path cost higher than current, fcost will also be higher
            if newPathCost >= pathCost.get(nextState, float('inf')):
                continue
                
            # Update costs
            pathCost[nextState] = newPathCost
            fCost[nextState] = aStarEvaluation(problem, action, pathCost[nextState], heuristic) 
            
            queue.push(action, fCost[nextState])
            path[action] = current
            
    

    # No path
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
