import collections
from typing import List

# Dinic algorithm for maximum flow
class Dinic:
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.graph = [[] for _ in range(n_nodes)]
        self.level = [-1] * n_nodes
        self.ptr = [0] * n_nodes # Pointer for DFS to avoid re-exploring edges

    def add_edge(self, u, v, capacity):
        self.graph[u].append([v, capacity, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1]) # Residual edge with 0 capacity

    def bfs(self, source, sink):
        self.level = [-1] * self.n
        self.level[source] = 0
        q = collections.deque([source])
        while q:
            u = q.popleft()
            for v, capacity, _ in self.graph[u]:
                if capacity > 0 and self.level[v] == -1:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[sink] != -1

    def dfs(self, u, sink, pushed):
        if pushed == 0:
            return 0
        if u == sink:
            return pushed
        
        while self.ptr[u] < len(self.graph[u]):
            edge = self.graph[u][self.ptr[u]]
            v, capacity, rev_idx = edge[0], edge[1], edge[2]

            if self.level[v] != self.level[u] + 1 or capacity == 0:
                self.ptr[u] += 1
                continue
            
            tr = self.dfs(v, sink, min(pushed, capacity))
            if tr == 0:
                self.ptr[u] += 1
                continue
            
            self.graph[u][self.ptr[u]][1] -= tr
            self.graph[v][rev_idx][1] += tr
            return tr
        return 0

    def max_flow(self, source, sink):
        flow = 0
        while self.bfs(source, sink):
            self.ptr = [0] * self.n # Reset ptr for each BFS phase
            while True:
                pushed = self.dfs(source, sink, float('inf'))
                if pushed == 0:
                    break
                flow += pushed
        return flow


class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        
        # Helper function to check if a given max_w is feasible
        def check(max_w):
            # If n=1, node 0 is vacuously reachable from all other (non-existent) nodes.
            # But constraint is 2 <= n. So this case won't be hit with valid input.
            if n == 1:
                return True 
            
            # Construct the flow network for Dinic
            # Nodes in flow network: S_F, T_F, u_in (0..n-1), u_out (n..2n-1)
            # S_F is 2*n, T_F is 2*n + 1
            num_flow_nodes = 2 * n + 2
            flow_solver = Dinic(num_flow_nodes)
            
            source_flow = 2 * n
            sink_flow = 2 * n + 1

            # 1. Edge (u_in, u_out) with capacity threshold
            # Node u_in maps to u, u_out maps to u + n.
            # These edges limit the outgoing degree of node 'u' in the original graph
            # (which translates to incoming degree of 'u' in the reversed graph).
            for u in range(n):
                flow_solver.add_edge(u, u + n, threshold) 

            # 2. For each eligible original edge (A, B, W) with W <= max_w:
            # Add an edge from (B)_out to (A)_in in the flow network.
            # This represents using the original edge A -> B, which in the reversed graph
            # means B can reach A.
            # (B)_out is B+n, (A)_in is A
            for u_orig, v_orig, w_orig in edges:
                if w_orig <= max_w:
                    # In reversed graph, edge is v_orig -> u_orig.
                    # Flow from v_orig_out (v_orig + n) to u_orig_in (u_orig)
                    flow_solver.add_edge(v_orig + n, u_orig, 1) 

            # 3. Source to 0_in: Node 0 must be reachable from all other nodes (in original graph).
            # This means 0 must be able to reach all other nodes in the reversed graph.
            # S_F pushes (n-1) units of flow into 0_in, indicating 0 needs to "reach" n-1 distinct nodes.
            # The bottleneck at (0_in, 0_out) will correctly limit node 0's outgoing degree.
            flow_solver.add_edge(source_flow, 0, n - 1) 

            # 4. For each u from 1 to n-1: add edge (u_out, T_F) with capacity 1
            # u_out is u + n. This consumes 1 unit of flow if node u is "reached".
            # We need to reach all n-1 nodes other than 0.
            for u in range(1, n):
                flow_solver.add_edge(u + n, sink_flow, 1) 

            # The total flow required is n-1 (one unit for each node from 1 to n-1).
            # If the max flow equals n-1, all conditions are satisfied for this max_w.
            return flow_solver.max_flow(source_flow, sink_flow) == n - 1

        # Binary search for the minimum max_w
        low = 1
        high = 10**6 # Max possible edge weight based on constraints
        ans = -1

        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                ans = mid
                high = mid - 1 # Try to find a smaller max_w
            else:
                low = mid + 1 # Need a larger max_w
        
        return ans