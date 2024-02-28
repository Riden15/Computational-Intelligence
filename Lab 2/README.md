# Nim Game
Nim is a game in which two players take turns removing objects from distinct rows. On each turn, a player must remove at least one object, and may remove any number of objects provided they all come from the same row. 

In our type of Nim, the player who wins is the one that has to take an object and only one is left. In this way, a player needs to play to force the opponent to take the last object.

## Evolved strategy
The main idea of our solution is to use some kind of probability to choose the best move.
- An individual is composed of N probabilities, where N is the number of rows in the nim game.
- The fitness function is out ``evaluate_population`` function. It returns the number of wins of the current population against the optimal strategy.
- With the ``make_move`` function we compute for each row the possible move that is (row,n object), where n object will
be a number between one and the actual number of objects in that row. Once we have the moves, we
choose the one which row corresponds to the highest probability of the individual.

Example: n = 5, individual = [0.1, 0.5, 0.4, 0.6, 0.9], moves = [(0,1), (1,1), (2,1), (3,4), (4,6)]
The individual tells us that the move (4,6) is the one with the highest prob.

## Collaboration
For this laboratory, I collaborate with [Nicholas Berardo s319441](https://github.com/Niiikkkk/Computational-Intelligence/tree/main)

