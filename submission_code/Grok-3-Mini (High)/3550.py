import math
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        INF = 10**15
        m = len(board)
        n = len(board[0])
        total_nodes = 2 + m + n
        s = 0
        t = 1
        row_start = 2
        col_start = 2 + m
        
        adj = [[] for _ in range(total_nodes)]
        edges = []
        
        def add_edge(u, v, cap, cost):
            idx = len(edges)
            edges.append([v, cap, cost, 0])  # [to, cap, cost, rev_idx]
            rev_idx = len(edges)
            edges.append([u, 0, -cost, idx])  # reverse edge
            edges[idx][3] = rev_idx
            edges[rev_idx][3] = idx
            adj[u].append(idx)
            adj[v].append(rev_idx)
        
        # Add edges from source to row nodes
        for i in range(m):
            add_edge(s, row_start + i, 1, 0)
        
        # Add edges from column nodes to sink
        for j in range(n):
            add_edge(col_start + j, t, 1, 0)
        
        # Add edges from row nodes to column nodes with negative weights
        for i in range(m):
            for j in range(n):
                cost = -board[i][j]
                add_edge(row_start + i, col_start + j, 1, cost)
        
        total_cost = 0
        
        # Add flow three times using successive shortest paths with Bellman-Ford
        for _ in range(3):
            dist = [INF] * total_nodes
            dist[s] = 0
            parent_node = [-1] * total_nodes
            parent_edge_idx = [-1] * total_nodes
            
            # Bellman-Ford algorithm
            for _ in range(total_nodes - 1):
                for u in range(total_nodes):
                    if dist[u] < INF:
                        for edge_idx in adj[u]:
                            edge = edges[edge_idx]
                            v = edge[0]
                            res_cap = edge[1]
                            if res_cap > 0 and dist[u] + edge[2] < dist[v]:
                                dist[v] = dist[u] + edge[2]
                                parent_node[v] = u
                                parent_edge_idx[v] = edge_idx
            
            # Check if there is a path to sink
            if dist[t] >= INF:
                # This should not happen as per constraints, but handle just in case
                return 0  # or raise an error, but return default for safety
            
            # Add the cost of the path
            total_cost += dist[t]
            
            # Augment flow along the path
            v_node = t
            while v_node != s:
                u_node = parent_node[v_node]
                edge_idx = parent_edge_idx[v_node]
                edges[edge_idx][1] -= 1  # Decrease residual capacity
                rev_idx = edges[edge_idx][3]
                edges[rev_idx][1] += 1  # Increase reverse residual capacity
                v_node = u_node
        
        # The maximum sum is the negative of the total minimum cost
        return -total_cost