import sys
from collections import deque
from typing import List

# Dinic algorithm implementation
class FlowNetwork:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj = [[] for _ in range(num_nodes)]
        self.level = [0] * num_nodes
        self.ptr = [0] * num_nodes

    def add_edge(self, u, v, capacity):
        # Store forward edge and backward (residual) edge
        self.adj[u].append([v, capacity, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1]) # Residual edge starting with 0 capacity

    def bfs(self, s, t):
        # Build level graph
        self.level = [-1] * self.num_nodes
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, capacity, _ in self.adj[u]:
                # If edge has capacity and v is not visited in this BFS layer
                if capacity > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        # Return True if sink is reachable from source in the level graph
        return self.level[t] != -1

    def dfs(self, u, t, pushed):
        # Find augmenting path in level graph and push flow
        if pushed == 0:
            return 0
        if u == t:
            return pushed
        # Iterate through adjacency list from saved pointer
        while self.ptr[u] < len(self.adj[u]):
            edge = self.adj[u][self.ptr[u]]
            v, capacity, rev_idx = edge
            
            # Check if v is in the next level and edge has capacity
            if self.level[v] != self.level[u] + 1 or capacity == 0:
                self.ptr[u] += 1
                continue
                
            # Recursively call dfs
            tr = self.dfs(v, t, min(pushed, capacity))
            
            # If flow is pushed through the recursive call
            if tr == 0:
                self.ptr[u] += 1
                continue
                
            # Update capacities
            edge[1] -= tr
            self.adj[v][rev_idx][1] += tr
            return tr
            
        # No augmenting path found from this node
        return 0

    def dinic(self, s, t):
        flow = 0
        # While sink is reachable from source in level graph
        while self.bfs(s, t):
            # Reset pointer for each node for new DFS phase
            self.ptr = [0] * self.num_nodes
            # Push flow until no more augmenting paths in the current level graph
            while True:
                pushed = self.dfs(s, t, float('inf')) # Try to push infinite flow
                if pushed == 0:
                    break
                flow += pushed
        return flow

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:

        # Check function using max flow
        # Returns True if it's possible to satisfy the conditions with max edge weight <= max_w
        def can_check(max_w):
            # Build the reversed graph edges with weight <= max_w
            # Store counts of parallel reversed edges
            # A reversed edge [v, u] exists if an original edge [u, v, w] exists
            # The constraint (out-degree on u in original) translates to (in-degree on u in reversed)
            rev_edges_counts = {} # Key: (v, u) tuple representing reversed edge v -> u; Value: count of original edges [u, v, w] with w <= max_w
            for u, v, w in edges:
                if w <= max_w:
                    # Reversed edge is v -> u
                    if (v, u) not in rev_edges_counts:
                        rev_edges_counts[(v, u)] = 0
                    rev_edges_counts[(v, u)] += 1

            # Build flow network for G_rev reachability from 0 with in-degree constraints on G_rev nodes
            # Flow network nodes: S, T, u_in, u_out for u = 0..n-1
            # S = 2n, T = 2n+1
            # u_in = 2u, u_out = 2u+1
            num_flow_nodes = 2 * n + 2
            s = 2 * n
            t = 2 * n + 1
            flow_net = FlowNetwork(num_flow_nodes)

            # Edge S -> 0_out (2n -> 2*0 + 1 = 1)
            # Capacity n-1 because n-1 other nodes (1 to n-1) need to be ultimately reached by flow originating conceptually from 0
            flow_net.add_edge(s, 2 * 0 + 1, n - 1)

            # Edge u_in -> u_out (2u -> 2u+1) with capacity threshold
            # Represents the in-degree constraint on node u in the selected reversed subgraph (G_rev_S)
            # The total flow entering u_in is limited by this capacity
            for u in range(n):
                flow_net.add_edge(2 * u, 2 * u + 1, threshold)

            # Edges v_out -> u_in (2v+1 -> 2u) for reversed edge [v, u] in G_rev_W
            # This flow edge represents using the reversed edge [v, u] in G_rev_S
            # Capacity is the number of original edges [u, v, w] with w <= max_w.
            # Using one such edge contributes 1 to the flow into u_in.
            for (v, u), count in rev_edges_counts.items():
                 flow_net.add_edge(2 * v + 1, 2 * u, count)

            # Edge u_out -> T (2u+1 -> 2n+1) with capacity 1 for u = 1..n-1
            # Represents the requirement for each node u != 0 to receive at least 1 unit of flow (be reached)
            # If the max flow reaches n-1, it means each node u=1..n-1 successfully received 1 unit of flow into T
            for u in range(1, n):
                flow_net.add_edge(2 * u + 1, t, 1)

            # Check if max flow is n-1
            # If max flow is n-1, it's possible to send flow from S to T such that
            # - n-1 units originate conceptually from 0 (S->0_out)
            # - Each node u=1..n-1 receives 1 unit into T (u_out->T)
            # - The total flow into each u_in (representing in-degree in G_rev_S) is <= threshold (u_in->u_out)
            # - This corresponds to existence of a subgraph in G_rev_W rooted at 0 covering all nodes with in-degree <= threshold
            return flow_net.dinic(s, t) == n - 1

        # Binary search for the minimum maximum weight
        # The possible range of maximum edge weight is [1, 10^6].
        # We search in the range [1, 10^6] inclusive.
        low = 1
        high = 10**6
        ans = -1 # Initialize answer to -1 (impossible case)

        while low <= high:
            mid = low + (high - low) // 2
            if can_check(mid):
                # Possible with max weight 'mid'. Store it and try smaller.
                ans = mid
                high = mid - mid + low -1 # Simplified: high = mid - 1
                high = mid - 1 # Correct logic

            else:
                # Not possible with max weight 'mid'. Need larger.
                low = mid + 1

        return ans