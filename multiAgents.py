# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from game import Actions
from itertools import product

class ReflexAgent(Agent):

    discountFactor = 0.5
    environmentVisibility = 2
    ghostSearchDepth = 0 #6
    ghostSearchPenalty = -20
    localFoodMap = []

    def getAction(self, gameState):
        # Update current world view
        curPos = gameState.getPacmanPosition()
        curFood = gameState.getFood()
        boardWidth = gameState.data.layout.width
        boardHeight = gameState.data.layout.height

        if not self.localFoodMap:
            self.localFoodMap = curFood
            for i in xrange(1, curFood.width):
                for j in xrange(1, curFood.height):
                    self.localFoodMap[i][j] = False


        for i in xrange(max(1, curPos[0] - self.environmentVisibility), min(boardWidth, curPos[0] + self.environmentVisibility + 1)):
            for j in xrange(max(1, curPos[1] - self.environmentVisibility), min(boardHeight, curPos[1] + self.environmentVisibility + 1)):
                self.localFoodMap[i][j] = curFood[i][j]

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        if bestScore == float("-inf"):
            return Directions.STOP

        return legalMoves[chosenIndex]

    def fillGhostFood(self, gameState, ghostIndex, foodMap, depth):
        if depth == 0:
            return

        for newGameState in [gameState.generateSuccessor(ghostIndex, action) for action in gameState.getLegalActions(ghostIndex)]:
            penalty = self.ghostSearchPenalty / (self.ghostSearchDepth - depth + 1)
            ghostPosition = newGameState.getGhostPosition(ghostIndex)
            if ghostPosition in foodMap:
                foodMap[ghostPosition] += penalty
            else:
                foodMap[ghostPosition] = penalty
            self.fillGhostFood(newGameState, ghostIndex, foodMap, depth - 1)
            del newGameState

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldPos = currentGameState.getPacmanPosition()
        newWall = successorGameState.getWalls()
        newGhostStates = successorGameState.getGhostStates()

        # Check if would run into a ghost with this action
        ghostFood = {}
        for i in xrange(len(newGhostStates)):
            ghostPosition = newGhostStates[i].getPosition()
            # If ghost not visible, don't calculate
            if not (abs(oldPos[0] - ghostPosition[0]) <= self.environmentVisibility >= abs(oldPos[1] - ghostPosition[1])):
                continue

            if newGhostStates[i].scaredTimer > 0:
                ghostFood[newGhostStates[i].getPosition()] = 200
                continue

            if successorGameState.getPacmanPosition() in  [Actions.getSuccessor(ghostPosition, ghostAction) for ghostAction in currentGameState.getLegalActions(i + 1)]:
                return float("-inf")

            self.fillGhostFood(currentGameState, i + 1, ghostFood, self.ghostSearchDepth)


        # Calculate how close is the rest of the food (BFS)
        discountedReward = 0
        vector = [int(v) for v in Actions.directionToVector(action)]
        if not vector == (0,0):
            visited = {}
            queue = [(newPos, 1)]
            while len(queue):
                (curPos, cost) = queue.pop(0)
                if curPos in visited:
                    continue

                visited[curPos] = 1
                if self.localFoodMap[curPos[0]][curPos[1]]:
                    discountedReward += 10.0 * self.discountFactor ** cost
                if curPos in ghostFood:
                    discountedReward += ghostFood[curPos] * self.discountFactor ** cost
                for expandedPos in Actions.getLegalNeighbors(curPos, newWall):
                    # Check if the area is visible and not calculated yet
                    if expandedPos not in visited:
                        queue.append((expandedPos, cost + 1))

        # Reward more for food that is in avaliable in the next move and isolated
        try:
            if currentGameState.hasFood(newPos[0], newPos[1]):
                for i in xrange(-1, 2):
                    for j in xrange(-1, 2):
                        if (i == 0 or j == 0) and not i == j:
                            if (0 < newPos[0] + i < currentGameState.data.layout.width) and (0 < newPos[1] + j < currentGameState.data.layout.height):
                                if self.localFoodMap[newPos[0] + i][ newPos[1] + j]:
                                    #print "Pacman: ", oldPos, "New pos:", newPos, "Food on:", (i, j)
                                    raise Exception
                    #print "Isolated food found"
                discountedReward *= 2
        except Exception:
            pass

        return scoreEvaluationFunction(successorGameState) + discountedReward

def scoreEvaluationFunction(currentGameState):
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


def system():
    pass


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        (score, action) = self.calculateMiniMax(gameState, self.index, self.depth)
        return action

    def calculateMiniMax(self, gameState, player, depth):
        if depth <= 0:
            return self.evaluationFunction(gameState), gameState.getPacmanState().getDirection()
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), gameState.getPacmanState().getDirection()

        if player == 0:
            max_val = float("-inf")
            max_action = Directions.STOP
            for action in gameState.getLegalActions(player):
                if action == Directions.STOP: continue
                #print "Pacman", action
                val, tmp = self.calculateMiniMax(gameState.generateSuccessor(player, action), 1, depth - 1)
                #print "Pacman", (val, action)
                if val >= max_val:
                    max_val = val
                    max_action = action
            return max_val, max_action
        else:
            args = []
            for i in xrange(1, gameState.getNumAgents()):
                args.append(gameState.getLegalActions(i))

            min_val = float("+inf")
            min_action = Directions.STOP
            for actionList in product(*args):
                newGameState = gameState
                for i in xrange(1, gameState.getNumAgents()):
                    #print (i, actionList[i - 1], gameState.getLegalActions(i), newGameState.getLegalActions(i))
                    if newGameState.isLose() or gameState.isWin():
                        return self.evaluationFunction(gameState), gameState.getPacmanState().getDirection()
                    newGameState = newGameState.generateSuccessor(i, actionList[i - 1])
                    #print "Ghost", actionList
                val, tmp = self.calculateMiniMax(newGameState, 0, depth - 1)
                #print "Ghost", (val, actionList)
                if val <= min_val:

                    min_val = val
                    min_action = actionList
            return min_val, min_action





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

