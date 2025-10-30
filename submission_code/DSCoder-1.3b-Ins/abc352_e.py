import sys
from heapq import *

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N+1)]
    indegree = [0] * (N+1)
    weights = [0] * (N+1)
    for _ in range(M):
        K, *A = map(int, sys.stdin.readline().split())
        for i in range(K):
            for j in range(i+1, K):
                graph[A[i]].append((A[j], weights[A[i]] + weights[A[j]]))
                graph[A[j]].append((A[i], weights[A[i]] + weights[A[j]]))
                indegree[A[j]] += 1
        weights[A[0]] += K

    if indegree[N] != M:
        print(-1)
    else:
        print(weights[N])

solve()