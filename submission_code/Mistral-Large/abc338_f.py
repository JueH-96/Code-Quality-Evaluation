import sys
import itertools

def find_minimum_walk(N, M, edges):
    # Create a distance matrix
    dist = [[float('inf')] * N for _ in range(N)]
    for u, v, w in edges:
        dist[u-1][v-1] = w

    # Floyd-Warshall algorithm to find all-pairs shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Try all permutations of vertices to find the minimum cost walk
    min_cost = float('inf')
    for perm in itertools.permutations(range(N)):
        cost = 0
        valid = True
        for i in range(N-1):
            if dist[perm[i]][perm[i+1]] == float('inf'):
                valid = False
                break
            cost += dist[perm[i]][perm[i+1]]
        if valid:
            min_cost = min(min_cost, cost)

    if min_cost == float('inf'):
        print("No")
    else:
        print(min_cost)

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    edges = []
    for _ in range(M):
        U = int(data[index])
        index += 1
        V = int(data[index])
        index += 1
        W = int(data[index])
        index += 1
        edges.append((U, V, W))

    find_minimum_walk(N, M, edges)

if __name__ == "__main__":
    main()