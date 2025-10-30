from typing import List
import heapq

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        # Build a flow network to find a min-cost flow of 3 units,
        # which corresponds to a maximum-weight matching of size 3.
        N = m + n + 2
        s = 0
        t = m + n + 1

        # Edge structure for MCMF
        class Edge:
            __slots__ = ("to", "rev", "cap", "cost")
            def __init__(self, to, rev, cap, cost):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.cost = cost

        graph = [[] for _ in range(N)]
        def add_edge(frm: int, to: int, cap: int, cost: int):
            graph[frm].append(Edge(to, len(graph[to]), cap, cost))
            graph[to].append(Edge(frm, len(graph[frm]) - 1, 0, -cost))

        # Source -> each row (capacity 1, cost 0)
        for i in range(m):
            add_edge(s, 1 + i, 1, 0)
        # Each column -> sink (capacity 1, cost 0)
        for j in range(n):
            add_edge(1 + m + j, t, 1, 0)
        # Row i -> Column j edges with cost = -board[i][j]
        for i in range(m):
            for j in range(n):
                add_edge(1 + i, 1 + m + j, 1, -board[i][j])

        # Min-cost max-flow (successive shortest augmenting path with potentials)
        INF = 10**30
        h = [0] * N    # potentials
        prevv = [0] * N
        preve = [0] * N

        flow = 0
        cost = 0
        while flow < 3:
            # Dijkstra to find shortest augmenting path in terms of reduced cost
            dist = [INF] * N
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, v = heapq.heappop(pq)
                if d > dist[v]:
                    continue
                for idx, e in enumerate(graph[v]):
                    if e.cap > 0:
                        nd = d + e.cost + h[v] - h[e.to]
                        if nd < dist[e.to]:
                            dist[e.to] = nd
                            prevv[e.to] = v
                            preve[e.to] = idx
                            heapq.heappush(pq, (nd, e.to))
            if dist[t] == INF:
                # cannot send more flow
                break
            # update potentials
            for v in range(N):
                if dist[v] < INF:
                    h[v] += dist[v]
            # send one unit of flow
            d = 1
            flow += d
            cost += d * h[t]
            v = t
            while v != s:
                e = graph[prevv[v]][preve[v]]
                e.cap -= d
                graph[v][e.rev].cap += d
                v = prevv[v]

        # cost is the minimum cost = - (maximum sum of values)
        return -cost