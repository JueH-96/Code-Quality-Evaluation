import sys

import threading
def main():
    import sys
    import collections

    sys.setrecursionlimit(1 << 25)
    N, T = map(int, sys.stdin.readline().split())
    A_list = []
    B_list = []
    for _ in range(T):
        Ai, Bi = map(int, sys.stdin.readline().split())
        A_list.append(Ai)
        B_list.append(Bi)

    player_scores = [0] * (N +1)  # players are 1-indexed
    counts = {}
    counts[0] = N

    for i in range(T):
        Ai = A_list[i]
        Bi = B_list[i]
        old_score = player_scores[Ai]
        new_score = old_score + Bi

        # Update counts
        counts[old_score] -=1
        if counts[old_score] == 0:
            del counts[old_score]

        counts[new_score] = counts.get(new_score, 0) +1

        player_scores[Ai] = new_score

        print(len(counts))
threading.Thread(target=main).start()