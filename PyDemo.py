import random


BLACK = 1       # BLACK first, if the last turn is BLACK, BLACK win!
WHITE = 0

def simulate(step, loops=1000):
    win = 0
    turn = BLACK

    for i in range(loops):
        for j in range(step):
            if j == step-1:
                if turn == BLACK:
                    win += 1
                turn = BLACK
            else:
                num = random.random()
                if num >= 1/5:          # if num<0.333, continue take. otherwise the oppenent takes!
                    turn = 1-turn

    return win/loops

if __name__ == '__main__':
    for i in range(1, 100):
        prob = simulate(i, loops=100000)
        print(i, ": ", prob)