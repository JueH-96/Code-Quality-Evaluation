from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # Collect all unique values from the grid
        unique_values = list({v for row in grid for v in row})
        value_to_idx = {v: idx for idx, v in enumerate(unique_values)}
        R = len(grid)
        V = len(unique_values)
        total_nodes = 1 + R + V + 1  # source, rows, values, sink
        source = 0
        sink = R + V + 1
        
        # Build the adjacency list for the graph
        adj = [[] for _ in range(total_nodes)]
        
        def add_edge(fr, to, cap, cost):
            # Add forward edge
            adj[fr].append((to, len(adj[to]), cap, cost))
            # Add reverse edge
            adj[to].append((fr, len(adj[fr]) - 1, 0, -cost))
        
        # Add edges from source to each row
        for row in range(R):
            row_node = row + 1
            add_edge(source, row_node, 1, 0)
        
        # Add edges from each row to its values
        for i in range(R):
            row_node = i + 1
            for val in grid[i]:
                value_idx = value_to_idx[val]
                value_node = R + 1 + value_idx
                add_edge(row_node, value_node, 1, -val)  # Use negative cost for minimization
        
        # Add edges from each value node to sink
        for value_idx in range(V):
            value_node = R + 1 + value_idx
            add_edge(value_node, sink, 1, 0)
        
        max_flow = 0
        min_cost = 0
        INF = float('inf')
        
        # Use Bellman-Ford to find shortest paths for min cost max flow
        while True:
            distance = [INF] * total_nodes
            prev_node = [-1] * total_nodes
            prev_edge = [-1] * total_nodes
            distance[source] = 0
            
            updated = True
            for _ in range(total_nodes - 1):
                updated = False
                for u in range(total_nodes):
                    if distance[u] == INF:
                        continue
                    for edge_idx, (v, rev, cap, cost) in enumerate(adj[u]):
                        if cap > 0 and distance[v] > distance[u] + cost:
                            distance[v] = distance[u] + cost
                            prev_node[v] = u
                            prev_edge[v] = edge_idx
                            updated = True
                if not updated:
                    break
            
            if distance[sink] == INF:
                break  # No more augmenting paths
            
            # Augment flow along the shortest path
            max_flow += 1
            min_cost += distance[sink]
            
            # Update residual capacities
            v = sink
            while v != source:
                u = prev_node[v]
                e_idx = prev_edge[v]
                # Update forward edge
                forward_edge = adj[u][e_idx]
                adj[u][e_idx] = (forward_edge[0], forward_edge[1], forward_edge[2] - 1, forward_edge[3])
                # Update reverse edge
                rev_e_idx = forward_edge[1]
                reverse_edge = adj[v][rev_e_idx]
                adj[v][rev_e_idx] = (reverse_edge[0], reverse_edge[1], reverse_edge[2] + 1, reverse_edge[3])
                v = u
        
        return -min_cost if max_flow > 0 else 0