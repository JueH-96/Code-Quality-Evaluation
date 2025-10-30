from typing import List
import math

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows_len = len(grid)
        if rows_len == 0:
            return 0
        
        # Find all distinct values and map to indices
        values_set = set()
        for row in grid:
            for val in row:
                values_set.add(val)
        values_list = list(values_set)
        num_values = len(values_set)
        value_to_idx = {val: idx for idx, val in enumerate(values_list)}
        
        # Nodes: 0: source, 1: sink, 2 to 2+rows_len-1: rows, 2+rows_len to 2+rows_len+num_values-1: values
        V = 2 + rows_len + num_values
        
        # Residual graph
        edges = []  # list of [to, cap, cost, rev_idx]
        adj = [[] for _ in range(V)]  # for each node, list of edge indices
        
        def add_edge(u, v, cap, cost):
            idx_f = len(edges)
            edges.append([v, cap, cost, 0])  # temporary rev_idx
            idx_r = len(edges)
            edges.append([u, 0, -cost, idx_f])  # reverse edge
            edges[idx_f][3] = idx_r
            edges[idx_r][3] = idx_f
            adj[u].append(idx_f)
            adj[v].append(idx_r)
        
        # Add edges from source to rows
        for r in range(rows_len):
            add_edge(0, 2 + r, 1, 0)
        
        # Add edges from values to sink
        for v_idx in range(num_values):
            add_edge(2 + rows_len + v_idx, 1, 1, 0)
        
        # Add edges from rows to values
        for r in range(rows_len):
            row_node = 2 + r
            row_values_set = set(grid[r])  # distinct values in row
            for val in row_values_set:
                v_idx = value_to_idx[val]
                value_node = 2 + rows_len + v_idx
                add_edge(row_node, value_node, 1, -val)  # cost -val
        
        # Min-cost flow
        flow_cost = 0
        while True:
            # Bellman-Ford
            dist = [float('inf')] * V
            dist[0] = 0
            parent_node = [-1] * V
            parent_edge_idx = [-1] * V
            
            for iter_count in range(V - 1):
                updated = False
                for u in range(V):
                    if dist[u] < float('inf'):
                        for idx in adj[u]:
                            edge = edges[idx]
                            v_to = edge[0]
                            if edge[1] > 0 and dist[u] + edge[2] < dist[v_to]:
                                dist[v_to] = dist[u] + edge[2]
                                parent_node[v_to] = u
                                parent_edge_idx[v_to] = idx
                                updated = True
                if not updated:
                    break
            
            if dist[1] == float('inf'):  # no path to sink
                break
            
            # Augment flow
            flow_cost += dist[1]
            # Trace back and update residual graph
            v_curr = 1  # sink
            while v_curr != 0:
                p_node = parent_node[v_curr]
                edge_idx = parent_edge_idx[v_curr]  # edge from p_node to v_curr
                edge = edges[edge_idx]
                # Decrease capacity forward
                edges[edge_idx][1] -= 1
                # Increase capacity reverse
                rev_idx = edge[3]
                edges[rev_idx][1] += 1
                v_curr = p_node  # go to parent node
            
        return -flow_cost