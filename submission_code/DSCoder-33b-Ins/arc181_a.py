# YOUR CODE HERE

import sys

def min_operations(P):
    N = len(P)
    P_inv = [0] * (N + 1)
    for i in range(N):
        P_inv[P[i]] = i + 1
    max_len = 0
    cur_len = 1
    for i in range(2, N + 1):
        if P_inv[i] > P_inv[i - 1]:
            cur_len += 1
        else:
            max_len = max(max_len, cur_len)
            cur_len = 1
    max_len = max(max_len, cur_len)
    return N - max_len

T = int(sys.stdin.readline())
for _ in range(T):
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    print(min_operations(P))