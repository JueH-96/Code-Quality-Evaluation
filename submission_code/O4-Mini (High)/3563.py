from typing import List
from collections import deque

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        m = len(grid)
        # Gather all unique values
        values = set()
        for row in grid:
            for v in row:
                values.add(v)
        unique = list(values)
        val_to_idx = {v: i for i, v in enumerate(unique)}
        n = len(unique)

        # Build flow network
        # Nodes: 0 = source, 1..m = rows, m+1..m+n = values, m+n+1 = sink
        N = 1 + m + n + 1
        source = 0
        sink = m + n + 1
        # Graph adjacency: g[u] = list of [to, cap, cost, rev_index]
        g = [[] for _ in range(N)]
        
        def add_edge(fr: int, to: int, cap: int, cost: int):
            # forward edge
            g[fr].append([to, cap, cost, len(g[to])])
            # reverse edge
            g[to].append([fr, 0, -cost, len(g[fr]) - 1])
        
        # source -> each row (cap=1, cost=0)
        # each row -> sink (skip) (cap=1, cost=0)
        for i in range(m):
            add_edge(source, 1 + i, 1, 0)
            add_edge(1 + i, sink,    1, 0)
        
        # row -> value edges (cap=1, cost = -value)
        for i, row in enumerate(grid):
            seen = set()
            for v in row:
                if v in seen:
                    continue
                seen.add(v)
                j = val_to_idx[v]
                add_edge(1 + i, 1 + m + j, 1, -v)
        
        # value -> sink (cap=1, cost=0)
        for j in range(n):
            add_edge(1 + m + j, sink, 1, 0)

        # Min-cost max-flow (successive SPFA)
        INF = 10**9
        flow = 0
        cost = 0
        # We want to send exactly m units of flow (one per row, skip or pick)
        while flow < m:
            dist = [INF] * N
            inq  = [False] * N
            prevv = [-1] * N
            preve = [-1] * N

            dist[source] = 0
            dq = deque([source])
            inq[source] = True

            # SPFA to find shortest path wrt cost
            while dq:
                u = dq.popleft()
                inq[u] = False
                for ei, e in enumerate(g[u]):
                    v, cap, w, rev = e
                    if cap > 0 and dist[v] > dist[u] + w:
                        dist[v] = dist[u] + w
                        prevv[v] = u
                        preve[v] = ei
                        if not inq[v]:
                            inq[v] = True
                            dq.append(v)

            if dist[sink] == INF:
                break

            # augment one unit of flow along that path
            d = 1  # all caps are 1 anyway
            v = sink
            while v != source:
                u = prevv[v]
                ei = preve[v]
                d = min(d, g[u][ei][1])
                v = u

            v = sink
            while v != source:
                u = prevv[v]
                ei = preve[v]
                e = g[u][ei]
                # decrease forward cap
                e[1] -= d
                # increase reverse cap
                g[v][e[3]][1] += d
                v = u

            flow += d
            cost += dist[sink] * d

        # cost is negative of total picked value sum
        return -cost