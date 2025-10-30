# YOUR CODE HERE
import sys
from itertools import combinations

def kruskal(N, edges, K):
    edges.sort(key=lambda x: x[2])
    parent = list(range(N + 1))
    rank = [0] * (N + 1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
            return True
        return False

    min_cost = float('inf')
    for edge_combination in combinations(edges, N - 1):
        parent = list(range(N + 1))
        rank = [0] * (N + 1)
        cost = 0
        for u, v, w in edge_combination:
            if union(u, v):
                cost += w
        if len(set(find(i) for i in range(1, N + 1))) == 1:
            min_cost = min(min_cost, cost % K)

    return min_cost

def main():
    import sys
    input = sys.stdin.read().split()
    index = 0
    N = int(input[index])
    M = int(input[index + 1])
    K = int(input[index + 2])
    index += 3
    edges = []
    for _ in range(M):
        u = int(input[index])
        v = int(input[index + 1])
        w = int(input[index + 2])
        edges.append((u, v, w))
        index += 3

    result = kruskal(N, edges, K)
    print(result)

if __name__ == "__main__":
    main()