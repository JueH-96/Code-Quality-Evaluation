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
        h = [0] * N  # potential
        prevv = [0] * N
        preve = [0] * N
        INF = float('inf')
        while maxf > 0:
            dist = [INF] * N
            inqueue = [False] * N
            dist[s] = 0
            from collections import deque
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
                return res
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
        values = set()
        for row in grid:
            values.update(row)
        if not values:
            return 0
        values = list(values)
        m = len(values)
        n = len(grid)
        
        value_to_node = {v: i + n + 1 for i, v in enumerate(values)}
        sink = n + m + 1
        total_nodes = sink + 1
        
        mcf = MinCostFlow(total_nodes)
        source = 0
        
        for i in range(n):
            row_node = i + 1
            mcf.add_edge(source, row_node, 1, 0)
            for val in grid[i]:
                val_node = value_to_node[val]
                mcf.add_edge(row_node, val_node, 1, -val)
        
        for val in values:
            val_node = value_to_node[val]
            mcf.add_edge(val_node, sink, 1, 0)
        
        max_flow = n
        cost = mcf.flow(source, sink, max_flow)
        return -cost