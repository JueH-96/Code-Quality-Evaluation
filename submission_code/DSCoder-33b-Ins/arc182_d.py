# YOUR CODE HERE

import sys

def solve(N, M, A, B):
    operations = 0
    for i in range(N):
        if A[i] != B[i]:
            diff = (B[i] - A[i]) % M
            if diff > M // 2:
                diff -= M
            operations += abs(diff)
            A[i] = (A[i] + diff) % M
            if i < N - 1 and A[i] == A[i + 1]:
                return -1
    return operations

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

print(solve(N, M, A, B))