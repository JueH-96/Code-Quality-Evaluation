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
            dist[s] = 0
            inqueue = [False] * N
            queue = []
            queue.append(s)
            inqueue[s] = True
            
            while queue:
                v = queue.pop(0)
                inqueue[v] = False
                for i, e in enumerate(self.graph[v]):
                    if e.capacity > 0 and dist[e.to] > dist[v] + e.cost + h[v] - h[e.to]:
                        dist[e.to] = dist[v] + e.cost + h[v] - h[e.to]
                        prevv[e.to] = v
                        preve[e.to] = i
                        if not inqueue[e.to]:
                            queue.append(e.to)
                            inqueue[e.to] = True
            if dist[t] == INF:
                return res  # No more augmenting paths
            
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
        if not grid:
            return 0
        
        R = len(grid)
        C = len(grid[0]) if R > 0 else 0
        
        # Extract all unique values
        values = set()
        for i in range(R):
            for j in range(C):
                values.add(grid[i][j])
        V = len(values)
        if V == 0:
            return 0
        
        # Assign node IDs
        source = 0
        rows = list(range(1, R + 1))
        value_nodes = {v: R + 1 + i for i, v in enumerate(values)}
        sink = R + V + 1
        
        # Create the MinCostFlow graph
        size = sink + 1
        mcf = MinCostFlow(size)
        
        # Add edges from source to each row
        for i in range(R):
            mcf.add_edge(source, i + 1, 1, 0)
        
        # Add edges from each row to its possible values
        for i in range(R):
            row = grid[i]
            for val in row:
                v_node = value_nodes[val]
                mcf.add_edge(i + 1, v_node, 1, -val)
        
        # Add edges from each value node to the sink
        for v in values:
            v_node = value_nodes[v]
            mcf.add_edge(v_node, sink, 1, 0)
        
        # Compute the maximum flow and the minimal cost
        max_flow = min(R, V)
        total_cost = mcf.flow(source, sink, max_flow)
        
        # The maximum sum is the negative of the total cost
        max_sum = -total_cost
        
        return max_sum