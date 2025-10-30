# YOUR CODE HERE
import sys
import threading

def main():
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    stones = dict(zip(X, A))
    positions = sorted(stones.keys())

    surplus = 0
    total_moves = 0
    prev_position = 0
    N = int(N)

    for p in positions:
        gap = p - prev_position - 1  # number of positions without stones between prev_position and p
        if surplus < gap:
            print(-1)
            return
        total_moves += gap  # need at least one move per empty position
        surplus -= gap  # use surplus to fill the gap
        surplus += stones[p] - 1  # update surplus with new stones at position p
        prev_position = p

    gap = N - prev_position  # positions after last stone
    if surplus < gap:
        print(-1)
        return
    total_moves += gap
    surplus -= gap
    if surplus != 0:
        print(-1)
        return

    print(total_moves)

threading.Thread(target=main).start()