# YOUR CODE HERE
import sys

def find_strongest_programmer(N, M, A, B):
    superiority = [set() for _ in range(N)]
    for i in range(M):
        superiority[A[i]-1].add(B[i]-1)
    for i in range(N):
        for j in range(N):
            if j in superiority[i]:
                superiority[j].update(superiority[i])
    strongest = [i for i in range(N) if len(superiority[i]) == N-1]
    if len(strongest) == 1:
        return strongest[0] + 1
    else:
        return -1

N, M = map(int, sys.stdin.readline().split())
A, B = [], []
for _ in range(M):
    a, b = map(int, sys.stdin.readline().split())
    A.append(a)
    B.append(b)
print(find_strongest_programmer(N, M, A, B))