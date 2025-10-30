import sys
import heapq

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, Q = read_ints()
    edges = []
    for i in range(Q):
        L, R, C = read_ints()
        for j in range(L, R + 1):
            edges.append((C, i + N, j))
    edges.sort()

    parent = list(range(N + Q + 1))
    rank = [0] * (N + Q + 1)

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

    cost = 0
    for C, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            cost += C

    if len(set(find(i) for i in range(1, N + Q + 1))) == 1:
        print(cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()