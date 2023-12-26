import pickle
import random
from game import Game, Move, Player
from tqdm.auto import tqdm


class RLPlayer2(Player):
    def __init__(self, player, esp_rate=0.3, lr=0.1) -> None:
        super().__init__()
        self.esp_rate = esp_rate
        self.lr = lr
        self.player = player
        self.states = []
        self.state_value = {}

    def add_state(self, s):
        self.states.append(s)

    def choose_action(self, game: Game) -> tuple[tuple[int, int], Move]:
        if (random.random() <= self.esp_rate):  # do exploration for 30% of the time
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            move = (from_pos, move)
        else:  # take the best one for 70% of the time
            pos = game.get_possible_moves(self.player)
            value_max = -999
            for p in pos:
                tmp = game.get_board()
                game.make_move(p[0], p[1])
                next_board = game.get_hash_board()
                game.set_board(tmp)
                value_act = 0 if self.state_value.get(next_board) is None else self.state_value.get(next_board)
                if value_act > value_max:
                    value_max = value_act
                    move = p
        return move

    def give_rew(self, reward):
        for v in reversed(self.states):
            if self.state_value.get(v) is None:
                self.state_value[v] = 0
            self.state_value[v] += self.lr * (reward - self.state_value[v])
            reward = self.state_value[v]


def train(player1, player2, game, epochs):
    for epoch in tqdm(range(epochs)):
        players = [player1, player2]
        current_player_idx = 1
        winner = -1
        while winner < 0:
            current_player_idx += 1
            current_player_idx %= len(players)
            ok = False
            while not ok:
                # from_pos is the position, for example [0,3]
                # slide is one element of Move (top,left...)
                from_pos, slide = players[current_player_idx].choose_action(game)
                ok = game.make_move(from_pos, slide)
            players[current_player_idx].add_state(game.get_hash_board())
            winner = game.check_winner()
            if winner >= 0:
                if winner == 0:
                    player1.give_rew(1)
                    player2.give_rew(0)
                else:
                    player1.give_rew(0)
                    player2.give_rew(1)
        game.reset()


def test(rl_player, random_player, num_games):
    g = Game()
    player2_wins = 0
    games = 0
    for _ in range(num_games):
        winner = g.play(rl_player, random_player)
        games += 1
        print(games)
        g.reset()
        if winner == 1:
            player2_wins += 1

    print(f"RLPlayer won {player2_wins / num_games * 100}%")
