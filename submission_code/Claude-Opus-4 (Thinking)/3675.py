class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        adj = defaultdict(list)
        
        # Build adjacency list
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # dp[u][0] = max sum when u can use k edges
        # dp[u][1] = max sum when u can use k-1 edges (parent edge is used)
        dp = [[0, 0] for _ in range(n)]
        
        def dfs(u, parent):
            children = []
            for v, w in adj[u]:
                if v != parent:
                    children.append((v, w))
            
            # Process all children first
            for v, w in children:
                dfs(v, u)
            
            # Calculate dp values for both cases
            for idx, limit in enumerate([k, k-1]):
                # For each child, calculate the gain of including its edge
                gains = []
                base_sum = 0
                
                for v, w in children:
                    # If we include edge (u,v), v has its parent edge used
                    include_val = w + dp[v][1]
                    # If we exclude edge (u,v), v has no parent edge
                    exclude_val = dp[v][0]
                    
                    gain = include_val - exclude_val
                    gains.append(gain)
                    base_sum += exclude_val
                
                # Sort gains in descending order and greedily select best edges
                gains.sort(reverse=True)
                
                result = base_sum
                for i in range(min(limit, len(gains))):
                    if gains[i] > 0:
                        result += gains[i]
                
                dp[u][idx] = result
        
        # Start DFS from node 0 as root
        dfs(0, -1)
        
        # Root has no parent edge, so it can use all k edges
        return dp[0][0]