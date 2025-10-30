# YOUR CODE HERE
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))

    pos = [0] * (N + 1)
    for i in range(N):
        pos[P[i]] = i

    min_diff = N
    for i in range(1, N - K + 2):
        max_pos = pos[i]
        for j in range(i, i + K):
            max_pos = max(max_pos, pos[j])
        min_diff = min(min_diff, max_pos - i + 1)

    print(min_diff)

solve()