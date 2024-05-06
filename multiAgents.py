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
import itertools

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
        CONST = 0.001
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = childGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        # Precalcular distancias de Manhattan
        foodList = newFood.asList()
        foodDistances = [manhattanDistance(
            newPos, foodPos) for foodPos in foodList]
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition(
        )) for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        scaredGhostDistances = [manhattanDistance(newPos, ghostState.getPosition(
        )) for ghostState in newGhostStates if ghostState.scaredTimer > 0]
        capsuleDistances = [manhattanDistance(
            newPos, capsulePos) for capsulePos in newCapsules]

        # Calcular distancias mínimas
        if not foodList:
            distToClosestFood = 0
        else:
            distToClosestFood = min(foodDistances)

        if not ghostDistances:
            distToClosestGhost = float("inf")
        else:
            distToClosestGhost = min(ghostDistances)

        if not any(newScaredTimes):
            scaredGhostTimer = 0
        else:
            scaredGhostTimer = sum(newScaredTimes)

        if not capsuleDistances:
            distToClosestCapsule = 0
        else:
            distToClosestCapsule = min(capsuleDistances)

        if not scaredGhostDistances:
            distToClosestScaredGhost = float("inf")
        else:
            distToClosestScaredGhost = min(scaredGhostDistances)

        # Calcular puntaje
        score = childGameState.getScore()
        # Acercarse a la comida y favorecer menos puntos de comida restantes
        score += 20 / (distToClosestFood + CONST) - 10 * len(foodList)
        if distToClosestGhost < 2:
            score -= 2000  # Huir del fantasma cercano
        else:
            # Alejarse de los fantasmas no asustados
            score -= 20 / (distToClosestGhost + CONST)
        score += scaredGhostTimer * 10  # Comer fantasmas asustados
        # Acercarse a las cápsulas
        score += 10 / (distToClosestCapsule + CONST)
        # Acercarse a los fantasmas asustados
        score += 20 / (distToClosestScaredGhost + CONST)

        return score


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
        """
        def miniMax(agentIndex, depth, gameState):
            # Base case: if game is won/lost or depth limit is reached, return the evaluation
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing player)
            if agentIndex == 0:  # 0 = Pacman, so we need to maximize the score
                return maxMove(agentIndex, depth, gameState)
            # Ghosts' turn (minimizing players)
            else:
                return minMove(agentIndex, depth, gameState)

        def maxMove(agentIndex, depth, gameState):
            maxEval = float("-inf")
            bestAction = None

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)

                # Call is recursive, cause when the agentIndex is greater than 0, it means that is a ghost turn
                eval = miniMax(1, depth, successorGameState)
                # So it will call the minMove function for the next agent until the ghosts have taken their turns

                if eval > maxEval:  # If the evaluation is greater than the current maxEval, update the maxEval and bestAction
                    maxEval = eval
                    bestAction = action

            if depth == 0:
                return bestAction  # Return the best action for Pacman
            else:
                return maxEval  # This is not the best action but the best evaluation, cause we are in a recursive call, and we need to continue with the next agent

        def minMove(agentIndex, depth, gameState):
            minEval = float("inf")  # The value as the largest possible value

            # Get the next agent index using modulo operator
            newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            # Explanation: If the current agent is the last agent, the next agent will be the first agent

            # Increase the depth if all agents have taken their turns
            # When 0 means is turn of Pacman so the depth is increased
            newDepth = depth if newAgentIndex > 0 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)
                minEval = min(miniMax(newAgentIndex, newDepth,
                              successorGameState), minEval)

            return minEval

        # Start the minimax process
        return miniMax(0, 0, gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)

    @References: https://www.researchgate.net/figure/The-Alpha-Beta-pseudo-code-from-Russell-and-Norvigs-AI-textbook-Russell-and-Norvig_fig1_329715244
    """

    def getAction(self, gameState):
        """
        Returns the alpha-beta pruning action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: if game is won/lost or depth limit is reached, return the evaluation
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                return maxMove(agentIndex, depth, gameState, alpha, beta)
            # Ghosts' turn (minimizing players)
            else:
                return minMove(agentIndex, depth, gameState, alpha, beta)

        def maxMove(agentIndex, depth, gameState, alpha, beta):
            maxEval = float("-inf")
            bestAction = None

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)
                eval = alphaBeta(1, depth, successorGameState, alpha, beta)
                if eval > maxEval:
                    maxEval = eval
                    bestAction = action
                if maxEval > beta:
                    return maxEval  # Beta pruning
                alpha = max(alpha, maxEval)

            if depth == 0:
                return bestAction
            else:
                return maxEval

        def minMove(agentIndex, depth, gameState, alpha, beta):
            minEval = float("inf")

            newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            newDepth = depth if newAgentIndex > 0 else depth + 1

            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.getNextState(agentIndex, action)
                eval = alphaBeta(newAgentIndex, newDepth,
                                 successorGameState, alpha, beta)
                minEval = min(minEval, eval)
                if minEval < alpha:
                    return minEval  # Alpha pruning
                beta = min(beta, minEval)

            return minEval

        alpha = float("-inf")
        beta = float("inf")
        return alphaBeta(0, 0, gameState, alpha, beta)


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
        # util.raiseNotDefined()
        action, score = self.get_value(gameState, 0, 0)

        return action

    def get_value(self, gameState, index, depth):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return "", self.evaluationFunction(gameState)

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(gameState, index, depth)

        # Expectation-agent: Ghost has index > 0
        else:
            return self.expected_value(gameState, index, depth)

    def max_value(self, gameState, index, depth):

        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            next_state = gameState.getNextState(index, action)
            next_state_index = index + 1
            next_state_depth = depth

            if next_state_index == gameState.getNumAgents():
                next_state_index = 0
                next_state_depth += 1

            current_action, current_value = self.get_value(
                next_state, next_state_index, next_state_depth)

            if current_value > max_value:
                max_value = current_value
                max_action = action

        return max_action, max_value

    def expected_value(self, gameState, index, depth):

        legalMoves = gameState.getLegalActions(index)
        expected_value = 0
        expected_action = ""

        next_state_probability = 1.0 / len(legalMoves)

        for action in legalMoves:
            next_state = gameState.getNextState(index, action)
            next_state_index = index + 1
            next_state_depth = depth

            if next_state_index == gameState.getNumAgents():
                next_state_index = 0
                next_state_depth += 1

            current_action, current_value = self.get_value(
                next_state, next_state_index, next_state_depth)

            expected_value += next_state_probability * current_value

        return expected_action, expected_value


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
