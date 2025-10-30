import collections

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n_rows = len(grid)
        distinct_vals = set()
        for row in grid:
            for val in row:
                distinct_vals.add(val)
        n_vals = len(distinct_vals)
        val_to_index = {val: idx for idx, val in enumerate(distinct_vals)}
        
        total_nodes = n_rows + n_vals + 2
        source = 0
        sink = total_nodes - 1
        
        graph = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, capacity, cost):
            reverse_idx1 = len(graph[v])
            graph[u].append([v, capacity, cost, reverse_idx1])
            reverse_idx2 = len(graph[u]) - 1
            graph[v].append([u, 0, -cost, reverse_idx2])
        
        for i in range(n_rows):
            add_edge(source, i + 1, 1, 0)
        
        for j in range(n_vals):
            add_edge(n_rows + 1 + j, sink, 1, 0)
        
        for i in range(n_rows):
            row_vals = set(grid[i])
            for val in row_vals:
                idx_val = val_to_index[val]
                add_edge(i + 1, n_rows + 1 + idx_val, 1, -val)
        
        INF = 10**18
        total_flow = 0
        total_cost = 0
        
        for _ in range(n_rows):
            dist = [INF] * total_nodes
            parent = [(-1, -1)] * total_nodes
            in_queue = [False] * total_nodes
            queue = collections.deque()
            
            dist[source] = 0
            queue.append(source)
            in_queue[source] = True
            
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for idx, edge in enumerate(graph[u]):
                    v, capacity, cost_val, rev_idx = edge
                    if capacity > 0 and dist[u] + cost_val < dist[v]:
                        dist[v] = dist[u] + cost_val
                        parent[v] = (u, idx)
                        if not in_queue[v]:
                            in_queue[v] = True
                            queue.append(v)
            if dist[sink] == INF:
                break
            flow_add = 1
            total_flow += flow_add
            cur = sink
            while cur != source:
                u, idx = parent[cur]
                edge = graph[u][idx]
                edge[1] -= flow_add
                rev_edge = graph[edge[0]][edge[3]]
                rev_edge[1] += flow_add
                total_cost += flow_add * edge[2]
                cur = u
        
        return -total_cost