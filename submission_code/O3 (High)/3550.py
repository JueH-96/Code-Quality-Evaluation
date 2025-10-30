from typing import List
from collections import deque

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        # ---------- min-cost max-flow template (SPFA based, enough because flow = 3) ----------
        class Edge:
            __slots__ = ("to", "rev", "cap", "cost")
            def __init__(self, to: int, rev: int, cap: int, cost: int):
                self.to = to          # destination node
                self.rev = rev        # index of the reverse edge in graph[to]
                self.cap = cap        # remaining capacity
                self.cost = cost      # cost per unit of flow

        def add_edge(g, fr, to, cap, cost):
            g[fr].append(Edge(to, len(g[to]), cap, cost))
            g[to].append(Edge(fr, len(g[fr]) - 1, 0, -cost))

        def min_cost_flow(g, s, t, max_flow):
            n = len(g)
            INF = 10**18
            flow = 0
            cost = 0
            dist = [0] * n
            prevv = [0] * n      # previous vertex in shortest path
            preve = [0] * n      # index of the edge used from prevv[v] to v

            while flow < max_flow:
                for i in range(n):
                    dist[i] = INF
                dist[s] = 0
                inq = [False] * n
                q = deque([s])
                inq[s] = True
                while q:
                    v = q.popleft()
                    inq[v] = False
                    for i, e in enumerate(g[v]):
                        if e.cap and dist[e.to] > dist[v] + e.cost:
                            dist[e.to] = dist[v] + e.cost
                            prevv[e.to] = v
                            preve[e.to] = i
                            if not inq[e.to]:
                                inq[e.to] = True
                                q.append(e.to)

                # no more augmenting path
                if dist[t] == INF:
                    break

                # bottleneck capacity on that path (will always be 1 here, but keep generic)
                d = max_flow - flow
                v = t
                while v != s:
                    d = min(d, g[prevv[v]][preve[v]].cap)
                    v = prevv[v]

                # augment
                v = t
                while v != s:
                    e = g[prevv[v]][preve[v]]
                    e.cap -= d
                    g[v][e.rev].cap += d
                    v = prevv[v]

                flow += d
                cost += d * dist[t]

            return cost  # minimal cost of sending `flow` units

        m = len(board)
        n = len(board[0])
        k = 3  # number of rooks to place

        # Build network
        # Node indexing:
        # 0           : super source
        # 1 .. m      : rows
        # m+1 .. m+n  : columns
        # m+n+1       : super sink
        V = m + n + 2
        S = 0
        T = V - 1
        graph = [[] for _ in range(V)]

        # edges from source to each row (capacity 1, cost 0)
        for r in range(m):
            add_edge(graph, S, 1 + r, 1, 0)

        # edges from each column to sink (capacity 1, cost 0)
        for c in range(n):
            add_edge(graph, 1 + m + c, T, 1, 0)

        # edges from rows to columns with cost = -board[r][c] (we minimize cost)
        for r in range(m):
            row_node = 1 + r
            for c in range(n):
                col_node = 1 + m + c
                add_edge(graph, row_node, col_node, 1, -board[r][c])

        # run min-cost flow for exactly 3 units
        min_cost = min_cost_flow(graph, S, T, k)

        # the maximal sum is the negative of the minimal cost obtained
        return -min_cost