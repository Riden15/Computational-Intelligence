from game import Player, Game, Move
import numpy as np
import random
import math


class MinMaxPlayer(Player):
    def __init__(self, playerPlaying, levels_depth):
        super().__init__()
        self.moves_value = []
        self.playerPlaying = playerPlaying
        self.levels_depth = levels_depth

    def game_evaluation(self, game: Game, depth):
        """Playing for X, if it wins, the reward is 1 otherwise is -1, the opposite when playing for O"""
        win = game.check_winner()
        ret = 0 + depth
        if win == 0:
            ret = 100 + depth
        elif win == 1:
            ret = -100 - depth
        return ret

    def min_max(self, game: Game, alpha, beta, depth):
        if depth <= 0 or game.check_winner() != -1:
            return self.game_evaluation(game, depth), None
        best_move = None
        if game.current_player == 0:
            for move in get_possible_moves(game, game.get_current_player()):
                tmp = game.get_board()
                g = Game()
                g.set_board(tmp)
                g.current_player = 0
                g.make_move(move[0], move[1])
                g.current_player = 1
                eval, _ = self.min_max(g, alpha, beta, depth - 1)
                if eval > alpha:
                    alpha = eval
                    best_move = move
                if beta <= alpha:
                    break
            return alpha, best_move
        else:
            for move in get_possible_moves(game, game.get_current_player()):
                tmp = game.get_board()
                g = Game()
                g.set_board(tmp)
                g.current_player = 1
                g.make_move(move[0], move[1])
                g.current_player = 0
                eval, _ = self.min_max(g, alpha, beta, depth - 1)
                if eval < beta:
                    beta = eval
                    best_move = move
                if beta <= alpha:
                    break
            return beta, best_move

        ### DOPO UN PO' NON MI CAMBIA PIù LA BOARD!!!!

    def choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        possibleMoves = get_possible_moves(game, game.get_current_player())
        possibleMoveCount = len(possibleMoves)

        depth = 0
        for depthLvl in self.levels_depth:
            if (possibleMoveCount > depthLvl[1]):
                depth = depthLvl[0]
            else:
                break

        _, move = self.min_max(game, -math.inf, math.inf, depth)

        return move

    def give_rew(self, reward):
        pass


def get_possible_moves(game: 'Game', player: int) -> list[tuple[tuple[int, int], Move]]:
    # possible moves:
    # - take border empty and fill the hole by moving in the 3 directions
    # - take one of your blocks on the border and fill the hole by moving in the 3 directions
    # 44 at start possible moves
    pos = set()
    for r in [0, 4]:
        for c in range(5):
            if game.get_board()[r, c] == -1 or game.get_board()[r, c] == player:
                if r == 0 and c == 0:  # OK
                    pos.add(((c, r), Move.BOTTOM))
                    pos.add(((c, r), Move.RIGHT))
                elif r == 0 and c == 4:  # OK
                    pos.add(((c, r), Move.BOTTOM))
                    pos.add(((c, r), Move.LEFT))
                elif r == 4 and c == 0:  # OK
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.RIGHT))
                elif r == 4 and c == 4:  # OK
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.LEFT))
                elif r == 0:  # OK
                    pos.add(((c, r), Move.BOTTOM))
                    pos.add(((c, r), Move.LEFT))
                    pos.add(((c, r), Move.RIGHT))
                elif r == 4:  # OK
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.LEFT))
                    pos.add(((c, r), Move.RIGHT))
    for c in [0, 4]:
        for r in range(5):
            if game.get_board()[r, c] == -1 or game.get_board()[r, c] == player:
                if r == 0 and c == 0:  # OK
                    pos.add(((c, r), Move.BOTTOM))
                    pos.add(((c, r), Move.RIGHT))
                elif r == 0 and c == 4:  # OK
                    pos.add(((c, r), Move.BOTTOM))
                    pos.add(((c, r), Move.LEFT))
                elif r == 4 and c == 0:  # OK
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.RIGHT))
                elif r == 4 and c == 4:  # OK
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.LEFT))
                elif c == 0:
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.RIGHT))
                    pos.add(((c, r), Move.BOTTOM))
                elif c == 4:
                    pos.add(((c, r), Move.TOP))
                    pos.add(((c, r), Move.LEFT))
                    pos.add(((c, r), Move.BOTTOM))
    return list(pos)
