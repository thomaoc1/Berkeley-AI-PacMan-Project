# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from typing import Optional

from learningAgents import ValueEstimationAgent
from mdp import MarkovDecisionProcess
import util


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.values = util.Counter()
        self.runValueIteration(iterations)

    def runValueIteration(self, iterations: int):
        for _ in range(iterations):
            for state in self.mdp.getStates():
                bestTotal = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    currentTotal = 0
                    for stateProbPair in self.mdp.getTransitionStatesAndProbs(state, action):
                        bestTotal += stateProbPair[1] \
                                    * (self.mdp.getReward(state, action, stateProbPair[0]) \
                                    + self.discount * self.values[stateProbPair[0]])
                    
                    bestTotal = max(currentTotal, bestTotal)
                
                if bestTotal == float('-inf'):
                    continue

                self.values[state] = bestTotal


    def getValue(self, state) -> float:
        """
          Return the value of the state (computed in __init__).
        """
        # TODO
        return self.values[state]

    def computeQValueFromValues(self, state, action) -> float:
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO
        qval = 0
        for stateProbPair in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += stateProbPair[1] \
                        * (self.mdp.getReward(state, action, stateProbPair[0]) \
                                + self.discount * self.values[stateProbPair[0]])
        return qval

    def computeActionFromValues(self, state) -> Optional[str]:
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO
        actions = self.mdp.getPossibleActions(state)
        maxaction = None
        maxvalueoveractions = -999999
        for action in actions: 
           value = self.computeQValueFromValues(state,action)
           if value > maxvalueoveractions:
                maxvalueoveractions = value
                maxaction = action  
        return maxaction
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)