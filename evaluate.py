from game2048.game import Game
from game2048.displays import Display
import tensorflow as tf


def single_run(size, score_to_win, AgentClass, sess):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), sess=sess)
    agent.build()   
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    '''====================
    Use your own agent here.'''
    from my_agent import MyAgent
    '''===================='''

    scores = []
    for cnt in range(N_TESTS):
        tf.reset_default_graph()
        sess = tf.Session()
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=MyAgent, sess=sess)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
