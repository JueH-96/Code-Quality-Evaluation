class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        def dfs(node, parent):
            # Get all children
            children = []
            for neighbor, weight in adj[node]:
                if neighbor != parent:
                    children.append((neighbor, weight))
            
            # Get DP values for all children
            child_results = []
            for child, weight in children:
                child_dp = dfs(child, node)
                child_results.append((child_dp, weight))
            
            # dp[0] = max weight if we don't take parent edge
            # dp[1] = max weight if we do take parent edge
            dp = [0, 0]
            
            # For each case (take parent edge or not)
            for take_parent in [0, 1]:
                remaining_capacity = k - take_parent
                
                # Calculate gain from taking each child edge
                gains = []
                base_value = 0
                for child_dp, weight in child_results:
                    gain = weight + child_dp[1] - child_dp[0]
                    gains.append(gain)
                    base_value += child_dp[0]
                
                # Take the best gains up to our capacity
                gains.sort(reverse=True)
                total = base_value
                for i in range(min(remaining_capacity, len(gains))):
                    if gains[i] > 0:
                        total += gains[i]
                
                dp[take_parent] = total
            
            return dp
        
        result = dfs(0, -1)
        return result[0]  # Root has no parent