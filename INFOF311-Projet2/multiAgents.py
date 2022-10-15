# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
from pacman import GameState
import util

from game import Agent



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, depth : int, gameState : GameState, agentIdx = 0):
        
        # Decrementing depth
        if agentIdx and not agentIdx % gameState.getNumAgents():
            depth -= 1
        
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return ([], self.evaluationFunction(gameState))
        
        # Rotating agents
        agentIdx %= gameState.getNumAgents()

        # Which initial optimal cost to use
        optmActionCost = float('-inf') if agentIdx == 0 else float('inf')
        optmAction = []
        
        # Minimising agent comparator
        lt = lambda cost1, cost2 : cost1 < cost2
        # Maximising agent comparator
        gt = lambda cost1, cost2 : cost1 > cost2
        compareCost = gt if agentIdx == 0 else lt

        # Minimax tree
        for action in gameState.getLegalActions(agentIdx):
            nextState = gameState.getNextState(agentIdx, action)
            actionCost = self.minimax(depth, nextState, agentIdx + 1)[1]
            if compareCost(actionCost, optmActionCost):
                optmActionCost, optmAction = actionCost, action 

        return (optmAction, optmActionCost)

    def getAction(self, state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(self.depth, state)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax(self, depth : int, gameState : GameState, agentIdx=0, alpha=float('-inf') , beta=float('inf')):
        
        # Decrementing depth
        if agentIdx and not agentIdx % gameState.getNumAgents():
            depth -= 1
        
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return ([], self.evaluationFunction(gameState))
        
        # Rotating agents
        agentIdx %= gameState.getNumAgents()

        # Which initial optimal cost to use
        optmActionCost = float('-inf') if agentIdx == 0 else float('inf')
        optmAction = []
        
        # Minimising agent comparator
        lt = lambda cost1, cost2 : cost1 < cost2
        # Maximising agent comparator
        gt = lambda cost1, cost2 : cost1 > cost2
        compareCost = gt if agentIdx == 0 else lt

        # Minimax tree
        for action in gameState.getLegalActions(agentIdx):
            nextState = gameState.getNextState(agentIdx, action)
            actionCost = self.minimax(depth, nextState, agentIdx + 1, alpha, beta)[1]
            
            if compareCost(actionCost, optmActionCost):
                optmActionCost, optmAction = actionCost, action 

            # Prunning
            if agentIdx == 0: alpha = max(alpha, actionCost)
            else: beta = min(beta, actionCost)

            if beta < alpha:
                break

        return (optmAction, optmActionCost)

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(self.depth, state)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """


    def chanceAgent(self, depth : int, gameState : GameState, agentIdx):
        actionsCost = 0
        optmAction = []

        for action in gameState.getLegalActions(agentIdx):
            nextState = gameState.getNextState(agentIdx, action)
            actionsCost += (1 / len(gameState.getLegalActions(agentIdx))) * self.expectiMinimax(depth, nextState, agentIdx + 1)[1]

        return (optmAction, actionsCost)

    def maxiAgent(self, depth : int, gameState : GameState, agentIdx=0):
        optmActionCost = float('-inf')
        optmAction = []

        for action in gameState.getLegalActions(agentIdx):
            nextState = gameState.getNextState(agentIdx, action)
            actionCost = self.expectiMinimax(depth, nextState, agentIdx + 1)[1]
            if actionCost > optmActionCost:
                optmActionCost, optmAction = actionCost, action 

        return (optmAction, optmActionCost)

    def expectiMinimax(self, depth : int, gameState : GameState, agentIdx = 0):
        
        # Decrementing depth
        if agentIdx and not agentIdx % gameState.getNumAgents():
            depth -= 1
        
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return ([], self.evaluationFunction(gameState))
        
        # Rotating agents
        agentIdx %= gameState.getNumAgents()

        if agentIdx == 0: return self.maxiAgent(depth, gameState)
        else: return self.chanceAgent(depth, gameState, agentIdx)


    def getAction(self, state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectiMinimax(self.depth, state)[0]

def betterEvaluationFunction(state: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()    

# Abbreviation
better = betterEvaluationFunction
