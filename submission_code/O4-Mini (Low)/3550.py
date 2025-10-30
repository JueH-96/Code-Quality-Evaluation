from typing import List
from collections import deque

class Edge:
    def __init__(self, v, cap, cost, rev):
        self.v = v        # endpoint
        self.cap = cap    # residual capacity
        self.cost = cost  # cost per unit flow
        self.rev = rev    # index of reverse edge in graph[v]

class MCMF:
    def __init__(self, N):
        self.N = N
        self.graph = [[] for _ in range(N)]

    def add_edge(self, u, v, cap, cost):
        # forward edge
        self.graph[u].append(Edge(v, cap, cost, len(self.graph[v])))
        # reverse edge
        self.graph[v].append(Edge(u, 0, -cost, len(self.graph[u]) - 1))

    # Compute min‐cost max‐flow from s to t up to maxf flow.
    # Returns (flow, cost)
    def min_cost_flow(self, s, t, maxf):
        N = self.N
        flow = 0
        cost = 0
        dist = [0]*N
        in_queue = [False]*N
        prevv = [0]*N
        preve = [0]*N

        while flow < maxf:
            # SPFA to find shortest augmenting path
            for i in range(N):
                dist[i] = float('inf')
                in_queue[i] = False
            dist[s] = 0
            queue = deque([s])
            in_queue[s] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for ei, e in enumerate(self.graph[u]):
                    if e.cap > 0 and dist[e.v] > dist[u] + e.cost:
                        dist[e.v] = dist[u] + e.cost
                        prevv[e.v] = u
                        preve[e.v] = ei
                        if not in_queue[e.v]:
                            queue.append(e.v)
                            in_queue[e.v] = True

            if dist[t] == float('inf'):
                # cannot send more flow
                break

            # add as much as possible (here unit flows)
            d = maxf - flow
            # backtrack to find minimum residual capacity
            v = t
            while v != s:
                u = prevv[v]
                e = self.graph[u][preve[v]]
                d = min(d, e.cap)
                v = u

            flow += d
            cost += d * dist[t]
            # update residual graph
            v = t
            while v != s:
                u = prevv[v]
                e = self.graph[u][preve[v]]
                e.cap -= d
                self.graph[v][e.rev].cap += d
                v = u

        return flow, cost

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        # build bipartite graph: rows 1..m, columns m+1..m+n
        N = m + n + 2
        SRC = 0
        SNK = m + n + 1
        mcmf = MCMF(N)
        # source to each row
        for i in range(m):
            mcmf.add_edge(SRC, 1 + i, 1, 0)
        # each column to sink
        for j in range(n):
            mcmf.add_edge(1 + m + j, SNK, 1, 0)
        # row i to col j with cost = -board[i][j]
        for i in range(m):
            for j in range(n):
                # capacity 1, cost = -value to maximize sum
                mcmf.add_edge(1 + i, 1 + m + j, 1, -board[i][j])

        # find min cost flow of size 3
        flow, neg_cost = mcmf.min_cost_flow(SRC, SNK, 3)
        # if flow < 3, no solution, but problem guarantees m,n >=3
        return -neg_cost