import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    A.sort()
    B.sort()
    
    i, j = 0, 0
    while i < N and j < M:
        if A[i] <= B[j]:
            i += 1
        j += 1
        if i == j:
            return A[i-1]
    return B[j-1] + 1 if j == M else A[i-1]

print(solve())