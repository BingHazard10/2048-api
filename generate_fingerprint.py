import json
import numpy as np
from game2048.game import Game
import tensorflow as tf


def generate_fingerprint(AgentClass, **kwargs):
    sess = tf.Session()
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(game=game, sess=sess)
    agent.build()

    trace = []
    num = len(board_json)
    for index, board in enumerate(board_json):
        print('{} left.'.format(num - index))
        game.board = np.array(board)
        direction = agent.step()
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter

    '''====================
    Use your own agent here.'''
    from my_agent import MyAgent as TestAgent
    '''===================='''

    fingerprint = generate_fingerprint(TestAgent)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
