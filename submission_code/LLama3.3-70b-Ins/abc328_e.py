import sys
from itertools import combinations

def read_input():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u-1, v-1, w))
    return N, M, K, edges

def is_spanning_tree(N, edges):
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
                if rank[root_x] == rank[root_y]:
                    rank[root_y] += 1

    for u, v, _ in edges:
        union(u, v)

    return len(set(find(i) for i in range(N))) == 1

def main():
    N, M, K, edges = read_input()
    min_cost = float('inf')
    for r in range(1 << M):
        selected_edges = [edges[i] for i in range(M) if (r & (1 << i))]
        if len(selected_edges) == N - 1 and is_spanning_tree(N, selected_edges):
            cost = sum(w for _, _, w in selected_edges) % K
            min_cost = min(min_cost, cost)
    print(min_cost)

if __name__ == '__main__':
    main()