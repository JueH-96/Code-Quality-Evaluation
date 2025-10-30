import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1

    edges = []
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        w = int(input[ptr]); ptr += 1
        edges.append((w, u, v))
    edges.sort()

    class DSU:
        def __init__(self, size):
            self.parent = list(range(size + 1))
            self.rank = [0] * (size + 1)
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        def union(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return False
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1
            return True

    adj = [[] for _ in range(N + 1)]
    dsu = DSU(N)
    for w, u, v in edges:
        if dsu.union(u, v):
            adj[u].append((v, w))
            adj[v].append((u, w))

    root = 1
    max_edge = [0] * (N + 1)
    visited = [False] * (N + 1)
    q = deque()
    q.append((root, 0))
    visited[root] = True
    max_edge[root] = 0

    while q:
        u, current_max = q.popleft()
        for v, w in adj[u]:
            if not visited[v]:
                new_max = max(current_max, w)
                max_edge[v] = new_max
                visited[v] = True
                q.append((v, new_max))

    A = list(map(int, input[ptr:ptr + K]))
    ptr += K
    B = list(map(int, input[ptr:ptr + K]))
    ptr += K

    a_values = [max_edge[a] for a in A]
    b_values = [max_edge[b] for b in B]

    a_values.sort()
    b_values.sort()

    total = 0
    for a, b in zip(a_values, b_values):
        total += max(a, b)

    print(total)

if __name__ == '__main__':
    main()