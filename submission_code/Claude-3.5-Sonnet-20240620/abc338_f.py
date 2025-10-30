# YOUR CODE HERE
import sys
from itertools import permutations

def read_input():
    N, M = map(int, input().split())
    edges = [list(map(int, input().split())) for _ in range(M)]
    return N, M, edges

def floyd_warshall(N, edges):
    dist = [[float('inf')] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u-1][v-1] = min(dist[u-1][v-1], w)
    
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def solve(N, M, edges):
    dist = floyd_warshall(N, edges)
    
    min_weight = float('inf')
    for perm in permutations(range(N)):
        weight = 0
        for i in range(N):
            if dist[perm[i]][perm[(i+1)%N]] == float('inf'):
                break
            weight += dist[perm[i]][perm[(i+1)%N]]
        else:
            min_weight = min(min_weight, weight)
    
    return min_weight if min_weight != float('inf') else "No"

N, M, edges = read_input()
result = solve(N, M, edges)
print(result)