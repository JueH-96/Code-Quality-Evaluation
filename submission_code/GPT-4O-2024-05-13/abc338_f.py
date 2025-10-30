# YOUR CODE HERE
import sys
from itertools import permutations

def floyd_warshall(n, edges):
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u-1][v-1] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] < float('inf') and dist[k][j] < float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        U = int(data[index])
        V = int(data[index + 1])
        W = int(data[index + 2])
        edges.append((U, V, W))
        index += 3
    
    dist = floyd_warshall(N, edges)
    
    min_cost = float('inf')
    found = False
    
    for perm in permutations(range(N)):
        cost = 0
        valid = True
        for i in range(N - 1):
            if dist[perm[i]][perm[i + 1]] == float('inf'):
                valid = False
                break
            cost += dist[perm[i]][perm[i + 1]]
        if valid:
            found = True
            min_cost = min(min_cost, cost)
    
    if found:
        print(min_cost)
    else:
        print("No")

if __name__ == "__main__":
    main()