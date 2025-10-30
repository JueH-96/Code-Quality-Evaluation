# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    occupied_set = set()
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        occupied_set.add((a, b))

    moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]

    attacked_set = set()
    for a,b in occupied_set:
        for dx, dy in moves:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                attacked_set.add((x,y))

    total_bad_positions = len(occupied_set.union(attacked_set))

    total_positions = N * N
    available_positions = total_positions - total_bad_positions

    print(available_positions)

threading.Thread(target=main).start()