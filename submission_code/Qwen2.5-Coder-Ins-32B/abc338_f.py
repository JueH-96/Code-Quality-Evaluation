# YOUR CODE HERE
import sys
import itertools

def floyd_warshall(n, edges):
    dist = [[float('inf')] * n for _ in range(n)]
    for u, v, w in edges:
        dist[u-1][v-1] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def tsp(dist, n):
    min_cost = float('inf')
    for perm in itertools.permutations(range(n)):
        cost = 0
        for i in range(n-1):
            cost += dist[perm[i]][perm[i+1]]
        cost += dist[perm[-1]][perm[0]]
        if cost < min_cost:
            min_cost = cost
    return min_cost

def main():
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    index += 2
    edges = []
    for _ in range(M):
        U = int(input[index])
        V = int(input[index + 1])
        W = int(input[index + 2])
        edges.append((U, V, W))
        index += 3

    dist = floyd_warshall(N, edges)
    min_cost = tsp(dist, N)

    if min_cost == float('inf'):
        print("No")
    else:
        print(min_cost)

if __name__ == "__main__":
    main()