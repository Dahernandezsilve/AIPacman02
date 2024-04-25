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
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return childGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)

    @References: https://www.researchgate.net/figure/MiniMax-Algorithm-Pseduo-Code-In-Fig-3-there-is-a-pseudo-code-for-NegaMax-algorithm_fig2_262672371
    """

    def getAction(self, gameState):
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
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                return maxMove(agentIndex, depth, gameState)
            # Ghosts' turn (minimizing players)
            else:
                return minMove(agentIndex, depth, gameState)

        def maxMove(agentIndex, depth, gameState):
            maxEval = float("-inf")
            bestAction = None

            newDepth = depth if agentIndex < gameState.getNumAgents() - 1 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)
                eval = minimax(1, newDepth, successorGameState)

                if eval > maxEval:
                    maxEval = eval
                    bestAction = action

            if depth == 0:
                return bestAction
            else:
                return maxEval

        def minMove(agentIndex, depth, gameState):
            minEval = float("inf")
            # Get the next agent index using modulo operator
            newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            # Explanation: If the current agent is the last agent, the next agent will be the first agent

            # Increase the depth if all agents have taken their turns
            newDepth = depth if newAgentIndex > 0 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)
                eval = minimax(newAgentIndex, newDepth, successorGameState)
                if eval < minEval:
                    minEval = eval

            return minEval

        # Start the minimax process
        return minimax(0, 0, gameState)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        References:
            - https://www.researchgate.net/figure/The-Alpha-Beta-pseudo-code-from-Russell-and-Norvigs-AI-textbook-Russell-and-Norvig_fig1_329715244

        Questions:
            - why this implementation uses depth
            - why this implementation uses .getLegalActions(0) and .getLegalActions(agentIndex) in the minValue function
        """
        def maxValue(state, alpha, beta, depth):
            """
            Returns the minimax action using self.depth and self.evaluationFunction
            """
            if depth == 0 or state.isWin() or state.isLose():
                return self.utility(state)

            v = float('-inf')

            # Get legal action of the PacMan (agentIndex=0)
            for action in state.getLegalActions():
                v = max(v, minValue(self.result(
                    state, action, 0), alpha, beta, depth, 1))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, alpha, beta, depth, agentIndex):
            if state.isWin() or state.isLose():
                return self.utility(state)

            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, maxValue(self.result(
                        state, action, agentIndex), alpha, beta, depth - 1))
                else:
                    v = min(v, minValue(self.result(state, action, agentIndex),
                            alpha, beta, depth, agentIndex + 1))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestValue = float('-inf')

        for action in gameState.getLegalActions(0):
            value = minValue(self.result(gameState, action, 0),
                             alpha, beta, self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
                alpha = max(alpha, bestValue)
        return bestAction

    def utility(self, state):
        return self.evaluationFunction(state)

    def result(self, state, action, agentIndex):
        return state.getNextState(agentIndex, action)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
