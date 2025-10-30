# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1000000)

    N, T, M = map(int, sys.stdin.readline().split())
    incompatible_pairs = set()
    for _ in range(M):
        Ai, Bi = map(int, sys.stdin.readline().split())
        incompatible_pairs.add((Ai - 1, Bi - 1))
        incompatible_pairs.add((Bi - 1, Ai - 1))

    count = 0

    N_players = N
    T_subsets = T

    assigned_subsets = [-1] * N_players

    def search(index, k):
        nonlocal count
        if index == N_players:
            if k == T_subsets:
                # assigned_subsets is a partition of N players into T subsets
                # Need to check if any incompatible pairs are in the same subset
                valid = True
                for Ai, Bi in incompatible_pairs:
                    if assigned_subsets[Ai] == assigned_subsets[Bi]:
                        valid = False
                        break
                if valid:
                    count +=1
            return
        for i in range(min(k, T_subsets)):
            assigned_subsets[index] = i
            search(index +1, k)
        if k < T_subsets:
            assigned_subsets[index] = k
            search(index +1, k + 1)

    search(0, 0)

    print(count)

threading.Thread(target=main).start()