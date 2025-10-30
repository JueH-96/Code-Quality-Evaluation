from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        nodes = m + n + 2
        source = 0
        sink = m + n + 1
        adj = [[] for _ in range(nodes)]
        
        # Add edges from source to rows
        for i in range(m):
            self.add_edge(source, i + 1, 1, 0, adj)
        
        # Add edges from rows to columns
        for i in range(m):
            row_node = i + 1
            for j in range(n):
                col_node = m + 1 + j
                cost = -board[i][j]
                self.add_edge(row_node, col_node, 1, cost, adj)
        
        # Add edges from columns to sink
        for j in range(n):
            col_node = m + 1 + j
            self.add_edge(col_node, sink, 1, 0, adj)
        
        # Compute min cost flow of 3 units
        total_cost = self.min_cost_flow(source, sink, 3, nodes, adj)
        return -total_cost
    
    def add_edge(self, u, v, cap, cost, adj):
        adj[u].append((v, cap, cost, len(adj[v])))
        adj[v].append((u, 0, -cost, len(adj[u]) - 1))
    
    def min_cost_flow(self, s, t, maxf, nodes, adj):
        res = 0
        h = [0] * nodes  # Potential
        INF = float('inf')
        
        for _ in range(maxf):
            dist = [INF] * nodes
            dist[s] = 0
            prevv = [-1] * nodes
            preve = [-1] * nodes
            
            # Bellman-Ford
            for _ in range(nodes - 1):
                for u in range(nodes):
                    if dist[u] == INF:
                        continue
                    for i, (v, cap, cost, rev) in enumerate(adj[u]):
                        if cap > 0 and dist[v] > dist[u] + cost + h[u] - h[v]:
                            dist[v] = dist[u] + cost + h[u] - h[v]
                            prevv[v] = u
                            preve[v] = i
            
            if dist[t] == INF:
                return -1  # Not possible under problem constraints
            
            # Update potentials
            for u in range(nodes):
                if dist[u] < INF:
                    h[u] += dist[u]
            
            # Find the minimum flow along the path
            d = maxf
            v = t
            while v != s:
                d = min(d, adj[prevv[v]][preve[v]][1])
                v = prevv[v]
            
            # Augment the flow
            v = t
            while v != s:
                e = adj[prevv[v]][preve[v]]
                adj[prevv[v]][preve[v]] = (e[0], e[1] - d, e[2], e[3])
                rev_e = adj[v][e[3]]
                adj[v][e[3]] = (rev_e[0], rev_e[1] + d, rev_e[2], rev_e[3])
                res += d * e[2]
                v = prevv[v]
        
        return res