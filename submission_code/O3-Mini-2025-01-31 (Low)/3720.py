from collections import deque
from typing import List

# We'll implement Dinic's algorithm for the max flow.
class Dinic:
    class Edge:
        __slots__ = "v", "cap", "rev"
        def __init__(self, v, cap, rev):
            self.v = v
            self.cap = cap
            self.rev = rev

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
    
    def add_edge(self, u, v, cap):
        self.adj[u].append(Dinic.Edge(v, cap, len(self.adj[v])))
        self.adj[v].append(Dinic.Edge(u, 0, len(self.adj[u]) - 1))
    
    def bfs(self, s, t, level):
        for i in range(len(level)):
            level[i] = -1
        q = deque()
        level[s] = 0
        q.append(s)
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if level[e.v] < 0 and e.cap:
                    level[e.v] = level[u] + 1
                    q.append(e.v)
        return level[t] >= 0
    
    def send_flow(self, u, flow, t, level, start):
        if u == t:
            return flow
        while start[u] < len(self.adj[u]):
            e = self.adj[u][start[u]]
            if level[e.v] == level[u] + 1 and e.cap:
                current_flow = self.send_flow(e.v, min(flow, e.cap), t, level, start)
                if current_flow:
                    e.cap -= current_flow
                    self.adj[e.v][e.rev].cap += current_flow
                    return current_flow
            start[u] += 1
        return 0
    
    def max_flow(self, s, t):
        if s == t:
            return 0
        total = 0
        level = [-1] * self.n
        # while there is an augmenting path
        while self.bfs(s, t, level):
            start = [0] * self.n
            while True:
                flow = self.send_flow(s, float('inf'), t, level, start)
                if not flow:
                    break
                total += flow
        return total


class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # binary search range from low to high, over edge weights.
        low, high = 1, max(w for _,_,w in edges)
        ans = -1
        
        # Pre-group edges by weight might be useful for efficiency,
        # but since edges <= 1e5, we filter each time.
        def feasible(maxW):
            # Build bipartite graph as described.
            # Left side: nodes 1..n-1 (we map them to 0..(n-2) internally)
            # Right side: nodes 0..n-1 (we map them to n-1 .. n-1+n) 
            # We'll create flow graph nodes:
            # S = 0
            # left side nodes: 1..(n-1)  (count = n-1)
            # right side nodes: n.. (n + n -1) (count = n)
            # T = n + n
            total_nodes = 1 + (n - 1) + n + 1  # S + left + right + T
            S = 0
            left_start = 1
            right_start = left_start + (n - 1)
            T = right_start + n
            
            flow = Dinic(total_nodes)
            
            # S to each left node (for every v in 1..n-1)
            for i in range(n - 1):
                flow.add_edge(S, left_start + i, 1)
            
            # For each candidate edge with weight <= maxW:
            # Only consider edges where source v is not 0,
            # because only nodes v != 0 require a chosen outgoing edge.
            #
            # An edge v -> u (original) goes from left node representing v to right node representing u.
            for a, b, w in edges:
                if w > maxW:
                    continue
                # For feasibility we only need assignment for nodes with v != 0.
                if a == 0:
                    continue
                # Map: left index for a is (a-1) (since a>=1) and right index for b is b.
                left_idx = left_start + (a - 1)
                right_idx = right_start + b
                flow.add_edge(left_idx, right_idx, 1)
            
            # Right nodes to T with capacity = threshold.
            for i in range(n):
                flow.add_edge(right_start + i, T, threshold)
            
            maxflow = flow.max_flow(S, T)
            return maxflow == n - 1
        
        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans

# For testing locally (you can remove this part before submitting)
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    n = 5
    edges = [[1,0,1],[2,0,2],[3,0,1],[4,3,1],[2,1,1]]
    threshold = 2
    print(sol.minMaxWeight(n, edges, threshold))  # Expected output: 1

    # Example 2:
    n = 5
    edges = [[0,1,1],[0,2,2],[0,3,1],[0,4,1],[1,2,1],[1,4,1]]
    threshold = 1
    print(sol.minMaxWeight(n, edges, threshold))  # Expected output: -1

    # Example 3:
    n = 5
    edges = [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[3,4,2],[4,0,1]]
    threshold = 1
    print(sol.minMaxWeight(n, edges, threshold))  # Expected output: 2

    # Example 4:
    n = 5
    edges = [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[4,0,1]]
    threshold = 1
    print(sol.minMaxWeight(n, edges, threshold))  # Expected output: -1