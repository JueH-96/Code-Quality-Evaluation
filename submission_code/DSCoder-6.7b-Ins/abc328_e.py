import sys

def read_input():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]
    return N, M, K, edges

def kruskal(N, M, K, edges):
    parent = list(range(N+1))
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[y] = x
    edges.sort(key=lambda x: x[2])
    total_cost = 0
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            total_cost = (total_cost + w) % K
    return total_cost

def main():
    N, M, K, edges = read_input()
    print(kruskal(N, M, K, edges))

if __name__ == "__main__":
    main()