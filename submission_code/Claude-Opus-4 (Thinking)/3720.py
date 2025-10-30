class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        from collections import deque
        
        def can_build_valid_graph(max_weight):
            # Build reverse graph with edges having weight <= max_weight
            reverse_graph = [[] for _ in range(n)]
            
            for a, b, w in edges:
                if w <= max_weight:
                    reverse_graph[b].append(a)
            
            # BFS from node 0 in reverse graph to check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            
            while queue:
                node = queue.popleft()
                for neighbor in reverse_graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        count += 1
                        queue.append(neighbor)
            
            # If not all nodes are reachable, return False
            if count < n:
                return False
            
            # If all nodes are reachable and threshold >= 1, we can build a valid tree
            return True
        
        # Get all unique weights and sort them
        weights = sorted(set(w for _, _, w in edges))
        
        # Binary search on the maximum edge weight
        left, right = 0, len(weights) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if can_build_valid_graph(weights[mid]):
                result = weights[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return result