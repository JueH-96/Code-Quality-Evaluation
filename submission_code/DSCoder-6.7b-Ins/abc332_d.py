import sys
from collections import deque

def min_operations():
    H, W = map(int, sys.stdin.readline().split())
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    B = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]

    operations = 0
    for i in range(H):
        for j in range(W-1):
            if A[i][j] > A[i][j+1]:
                A[i][j], A[i][j+1] = A[i][j+1], A[i][j]
                operations += 1

    for i in range(H-1):
        for j in range(W):
            if A[i][j] > A[i+1][j]:
                A[i][j], A[i+1][j] = A[i+1][j], A[i][j]
                operations += 1

    if A == B:
        print(operations)
    else:
        print(-1)

min_operations()