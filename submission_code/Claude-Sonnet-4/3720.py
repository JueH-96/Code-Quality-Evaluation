class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        from collections import defaultdict, deque
        
        def canReachZero(maxWeight):
            # Build graph with edges having weight <= maxWeight
            graph = defaultdict(list)
            for a, b, w in edges:
                if w <= maxWeight:
                    graph[a].append((b, w))
            
            # For each node, keep only the threshold lightest outgoing edges
            for node in graph:
                if len(graph[node]) > threshold:
                    graph[node].sort(key=lambda x: x[1])
                    graph[node] = graph[node][:threshold]
            
            # Check if node 0 is reachable from all other nodes
            # We'll do BFS from each node to see if it can reach 0
            def canReach(start, target):
                if start == target:
                    return True
                visited = set()
                queue = deque([start])
                visited.add(start)
                
                while queue:
                    node = queue.popleft()
                    for neighbor, _ in graph[node]:
                        if neighbor == target:
                            return True
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                return False
            
            # Check if all nodes can reach node 0
            for i in range(1, n):
                if not canReach(i, 0):
                    return False
            return True
        
        # Get all unique weights and sort them
        weights = sorted(set(w for _, _, w in edges))
        
        left, right = 0, len(weights) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canReachZero(weights[mid]):
                result = weights[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return result