from typing import List
from collections import deque

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        # Build a flow graph for maximum‐weight 3‐matching between rows and columns
        # We'll do min‐cost max‐flow with cost = -value so that minimizing cost
        # corresponds to maximizing the sum of chosen cells.
        N = m + n + 2
        s = 0
        t = m + n + 1
        # adjacency list of edges
        graph = [[] for _ in range(N)]

        class Edge:
            __slots__ = ('to', 'rev', 'cap', 'cost')
            def __init__(self, to, rev, cap, cost):
                self.to = to        # endpoint
                self.rev = rev      # index of reverse edge in graph[to]
                self.cap = cap      # remaining capacity
                self.cost = cost    # cost of the edge

        def add_edge(fr: int, to: int, cap: int, cost: int):
            # forward edge
            graph[fr].append(Edge(to, len(graph[to]), cap, cost))
            # reverse edge
            graph[to].append(Edge(fr, len(graph[fr]) - 1, 0, -cost))

        # source -> each row (capacity 1, cost 0)
        for i in range(m):
            add_edge(s, 1 + i, 1, 0)
        # each row -> each column (capacity 1, cost = -board[i][j])
        for i in range(m):
            for j in range(n):
                add_edge(1 + i, 1 + m + j, 1, -board[i][j])
        # each column -> sink (capacity 1, cost 0)
        for j in range(n):
            add_edge(1 + m + j, t, 1, 0)

        flow = 0
        cost = 0
        maxf = 3
        INF = 10**30

        # We'll run SPFA up to 3 times to find augmenting paths of minimum cost
        while flow < maxf:
            dist = [INF] * N
            inqueue = [False] * N
            prevv = [0] * N   # previous vertex
            preve = [0] * N   # which edge we took

            dist[s] = 0
            dq = deque([s])
            inqueue[s] = True

            # SPFA to find shortest path in cost from s to t
            while dq:
                v = dq.popleft()
                inqueue[v] = False
                for ei, e in enumerate(graph[v]):
                    if e.cap > 0 and dist[e.to] > dist[v] + e.cost:
                        dist[e.to] = dist[v] + e.cost
                        prevv[e.to] = v
                        preve[e.to] = ei
                        if not inqueue[e.to]:
                            dq.append(e.to)
                            inqueue[e.to] = True

            # if we cannot reach t anymore, break
            if dist[t] == INF:
                break

            # find the bottleneck capacity on the path (will be 1 in our construction)
            d = maxf - flow
            v = t
            while v != s:
                e = graph[prevv[v]][preve[v]]
                d = min(d, e.cap)
                v = prevv[v]

            # apply the flow
            flow += d
            cost += dist[t] * d
            v = t
            while v != s:
                e = graph[prevv[v]][preve[v]]
                e.cap -= d
                graph[v][e.rev].cap += d
                v = prevv[v]

        # cost is the sum of (-board[i][j]) over chosen edges; negate it back
        return -cost