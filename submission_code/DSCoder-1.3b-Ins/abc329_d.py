# YOUR CODE HERE

import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    counts = [0] * (N + 1)
    for i in range(M):
        counts[A[i]] += 1

    winner = max(counts)
    for i in range(M):
        if counts[A[i]] == winner:
            print(A[i])
            break

if __name__ == "__main__":
    solve()