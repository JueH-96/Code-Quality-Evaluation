class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        total_nodes = m + n + 2
        source = 0
        sink = total_nodes - 1
        graph = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap, cost):
            forward_index = len(graph[v])
            graph[u].append([v, cap, cost, forward_index])
            reverse_index = len(graph[u]) - 1
            graph[v].append([u, 0, -cost, reverse_index])
        
        for i in range(m):
            add_edge(source, i + 1, 1, 0)
        
        for i in range(m):
            for j in range(n):
                add_edge(i + 1, m + 1 + j, 1, -board[i][j])
        
        for j in range(n):
            add_edge(m + 1 + j, sink, 1, 0)
        
        INF = 10**18
        total_flow = 0
        total_cost = 0
        
        for _ in range(3):
            dist = [INF] * total_nodes
            parent_node = [-1] * total_nodes
            parent_edge_index = [-1] * total_nodes
            dist[source] = 0
            updated = True
            for _ in range(total_nodes - 1):
                updated = False
                for u in range(total_nodes):
                    if dist[u] == INF:
                        continue
                    for idx, edge in enumerate(graph[u]):
                        v, cap, cost, _ = edge
                        if cap > 0 and dist[u] + cost < dist[v]:
                            dist[v] = dist[u] + cost
                            parent_node[v] = u
                            parent_edge_index[v] = idx
                            updated = True
                if not updated:
                    break
            if dist[sink] == INF:
                break
            total_flow += 1
            total_cost += dist[sink]
            cur = sink
            while cur != source:
                u = parent_node[cur]
                idx = parent_edge_index[cur]
                forward_edge = graph[u][idx]
                forward_edge[1] -= 1
                rev_idx = forward_edge[3]
                reverse_edge = graph[cur][rev_idx]
                reverse_edge[1] += 1
                cur = u
        
        return -total_cost