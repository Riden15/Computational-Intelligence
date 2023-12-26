import random
from game import Game, Move, Player
from QL_Player import QLPlayer, train, test
from RL_Player import RLPlayer, train, test
from test import RLPlayer2, train, test

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def choose_action(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

    def give_rew(self, reward):
        pass

    def add_state(self, s):
        pass


if __name__ == '__main__':
    g = Game()

    # Q-Learning
    alpha = 0.1
    epsilon = 0.3
    discount_factor = 0.9
    epochs = 10000
    num_test_games = 1000

    #player1 = QLPlayer(player=0, alpha=alpha, epsilon=epsilon, discount_factor=discount_factor)
    #player2 = QLPlayer(player=1, alpha=alpha, epsilon=epsilon, discount_factor=discount_factor)
    #Trainer_player1, Trained_player2 = train(player1, player2, g, epochs=epochs)

    #player1.save_policy('player1_QL')
    #player2.save_policy('player2_QL')

    #player1.load_policy('player1_QL')
    #random_player = RandomPlayer()
    #test(player1, random_player, num_test_games)

    # Reinforcement Learning
    #player1 = RLPlayer(player=0, alpha=alpha, epsilon=epsilon, discount_factor=discount_factor)
    #player2 = RLPlayer(player=1, alpha=alpha, epsilon=epsilon, discount_factor=discount_factor)
    #Trainer_player1, Trained_player2 = train(player1, player2, g, epochs=epochs)

    #player1.save_policy('player1_RL')
    #player2.save_policy('player2_RL')

    #player1.load_policy('player1_QL')
    random_player = RandomPlayer()
    #test(player1, random_player, num_test_games)

    player1 = RLPlayer2(player=0)
    player2 = RLPlayer2(player=1)
    train(player1, player2, g, epochs=epochs)

    test(random_player, player2, num_test_games)
