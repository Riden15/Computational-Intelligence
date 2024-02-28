# Quixo
This project is a game of Quixo, a board game similar to Tic-Tac-Toe. The game is played on a 5x5 board, and the goal 
is to get five of your pieces in a row. The catch is that you can only move the pieces on the outside of the board, and you
can only move them in a straight line. The game is played by two players, and each player has five pieces. The game is 
played by moving a piece to an empty space, and then pushing the row or column of the piece in the direction of the move.

## Files

- `game.py`: This file is used to play the game using the different players.
- `CustomGameClass.py`: This file contains the custom class that extend ``game.py``.
- `Quixo.ipynb`: This notebook contains the players with their testing.
- `Players.rar`: This file contains the trained players (RL_Player_1, RL_Player_2, RL_Player_3).

## Players

We develop two players:
- Reinforcement Learning player trained against a random player.
- Minimax player

We tried a lot of different parameters for the Reinforcement Learning player, but we saved the best three that are saved in the `Players.rar` file.
The hyperparameters of all the players are reported in the notebook.

## Resources used
- [Reinforcement Learning: an Easy Introduction to Value Iteration](https://towardsdatascience.com/reinforcement-learning-an-easy-introduction-to-value-iteration-e4cfe0731fd5#dab6)
- [Temporal difference learning](https://www.lancaster.ac.uk/stor-i-student-sites/jordan-j-hood/2021/04/12/reinforcement-learning-temporal-difference-td-learning/)

## Collaboration
For this project, I collaborate with [Nicholas Berardo s319441](https://github.com/Niiikkkk/Computational-Intelligence/tree/main)