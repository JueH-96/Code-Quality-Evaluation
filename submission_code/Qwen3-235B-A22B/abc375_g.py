import sys
import heapq
from collections import defaultdict, deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    edges = []
    adj = [[] for _ in range(N + 1)]
    for idx in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        edges.append((a, b, c))
        adj[a].append((b, c, idx))
        adj[b].append((a, c, idx))

    def dijkstra(start):
        dist = [float('inf')] * (N + 1)
        dist[start] = 0
        hq = [(0, start)]
        heapq.heapify(hq)
        while hq:
            d, u = heapq.heappop(hq)
            if d > dist[u]:
                continue
            for v, c, _ in adj[u]:
                if dist[v] > d + c:
                    dist[v] = d + c
                    heapq.heappush(hq, (dist[v], v))
        return dist

    d1 = dijkstra(1)
    dN = dijkstra(N)
    d = d1[N]

    is_spg = [False] * M
    spg_adj = [[] for _ in range(N + 1)]
    for idx in range(M):
        a, b, c = edges[idx]
        if d1[a] + c + dN[b] == d or d1[b] + c + dN[a] == d:
            is_spg[idx] = True
            spg_adj[a].append((b, c, idx))
            spg_adj[b].append((a, c, idx))

    bridges = set()
    disc = [0] * (N + 1)
    low = [0] * (N + 1)
    visited = [False] * (N + 1)
    time = 1

    def tarjan(u, parent_edge_idx):
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        visited[u] = True
        for (v, c, edge_idx) in spg_adj[u]:
            if edge_idx == parent_edge_idx:
                continue
            if not visited[v]:
                tarjan(v, edge_idx)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add(edge_idx)
            else:
                low[u] = min(low[u], disc[v])

    for u in range(1, N + 1):
        if not visited[u] and spg_adj[u]:
            tarjan(u, -1)

    class DSU:
        def __init__(self, size):
            self.parent = list(range(N + 2))
            self.rank = [0] * (N + 2)

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def unite(self, x, y):
            x_root = self.find(x)
            y_root = self.find(y)
            if x_root == y_root:
                return
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1

    dsu = DSU(N)
    for idx in range(M):
        if is_spg[idx] and (idx not in bridges):
            a, b, c = edges[idx]
            dsu.unite(a, b)

    component_1 = dsu.find(1)
    component_N = dsu.find(N)
    critical_bridges = set()

    if component_1 != component_N:
        bridge_tree = defaultdict(list)
        for idx in bridges:
            a, b, c = edges[idx]
            cu = dsu.find(a)
            cv = dsu.find(b)
            bridge_tree[cu].append(cv)
            bridge_tree[cv].append(cu)

        parents = {}
        queue = deque([component_1])
        parents[component_1] = None
        found = False

        while queue and not found:
            current = queue.popleft()
            for neighbor in bridge_tree.get(current, []):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == component_N:
                        found = True
                        break

        if found:
            path_edges = set()
            current = component_N
            while parents.get(current) is not None:
                p = parents[current]
                path_edges.add((current, p))
                path_edges.add((p, current))
                current = p

            for idx in bridges:
                if not is_spg[idx]:
                    continue
                a, b, c = edges[idx]
                cu = dsu.find(a)
                cv = dsu.find(b)
                if (cu, cv) in path_edges or (cv, cu) in path_edges:
                    critical_bridges.add(idx)

    for idx in range(M):
        if is_spg[idx] and (idx in critical_bridges):
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()