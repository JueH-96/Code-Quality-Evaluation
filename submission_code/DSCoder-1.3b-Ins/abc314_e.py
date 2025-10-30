import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    C = [0] * N
    P = [0] * N
    S = [0] * (N * P[0])
    for i in range(N):
        C[i], P[i] = map(int, sys.stdin.readline().split())
        for j in range(P[i]):
            S[i * P[i] + j] = int(sys.stdin.readline())
    # YOUR CODE HERE

solve()