class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        def can_reach_zero(adj, visited, node):
            if node == 0:
                return True
            visited[node] = True
            for next_node in adj[node]:
                if not visited[next_node] and can_reach_zero(adj, visited, next_node):
                    return True
            return False
        
        def is_valid_configuration(max_weight):
            # Create adjacency list with edges having weight <= max_weight
            adj = [[] for _ in range(n)]
            outgoing = [0] * n
            
            for u, v, w in edges:
                if w <= max_weight:
                    adj[u].append(v)
                    outgoing[u] += 1
            
            # Check if any node exceeds threshold outgoing edges
            for count in outgoing:
                if count > threshold:
                    return False
            
            # Check if all nodes can reach node 0
            for node in range(n):
                if node != 0:
                    visited = [False] * n
                    if not can_reach_zero(adj, visited, node):
                        return False
            
            return True
        
        # Binary search on the maximum edge weight
        left = 1
        right = max(w for _, _, w in edges)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if is_valid_configuration(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result