from typing import List
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity, cost):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost

class MinCostFlow:
    def __init__(self, N):
        self.N = N
        self.graph = [[] for _ in range(N)]
    
    def add_edge(self, fr, to, capacity, cost):
        forward = Edge(to, len(self.graph[to]), capacity, cost)
        backward = Edge(fr, len(self.graph[fr]), 0, -cost)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)
    
    def flow(self, s, t, maxf):
        N = self.N
        res = 0
        h = [0] * N  # Potential
        prevv = [0] * N
        preve = [0] * N
        INF = float('inf')
        
        while maxf > 0:
            dist = [INF] * N
            inqueue = [False] * N
            dist[s] = 0
            q = deque()
            q.append(s)
            inqueue[s] = True
            while q:
                v = q.popleft()
                inqueue[v] = False
                for i, e in enumerate(self.graph[v]):
                    if e.capacity > 0 and dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]:
                        dist[e.to] = dist[v] + e.cost + h[v] - h[e.to]
                        prevv[e.to] = v
                        preve[e.to] = i
                        if not inqueue[e.to]:
                            q.append(e.to)
                            inqueue[e.to] = True
            if dist[t] == INF:
                return -1  # No solution
            for v in range(N):
                h[v] += dist[v] if dist[v] < INF else 0
            
            d = maxf
            v = t
            while v != s:
                d = min(d, self.graph[prevv[v]][preve[v]].capacity)
                v = prevv[v]
            maxf -= d
            res += d * h[t]
            v = t
            while v != s:
                e = self.graph[prevv[v]][preve[v]]
                e.capacity -= d
                self.graph[v][e.rev].capacity += d
                v = prevv[v]
        return res

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 0:
            return 0
        
        total_nodes = n + 100 + 3  # source, rows, values, dummy, sink
        mcf = MinCostFlow(total_nodes)
        source = 0
        sink = n + 100 + 2
        dummy = n + 100 + 1
        
        for i in range(n):
            mcf.add_edge(source, i + 1, 1, 0)
            for v in grid[i]:
                value_node = n + v
                mcf.add_edge(i + 1, value_node, 1, -v)
            mcf.add_edge(i + 1, dummy, 1, 0)
        
        for v in range(1, 101):
            value_node = n + v
            mcf.add_edge(value_node, sink, 1, 0)
        
        mcf.add_edge(dummy, sink, n, 0)
        
        cost = mcf.flow(source, sink, n)
        return -cost