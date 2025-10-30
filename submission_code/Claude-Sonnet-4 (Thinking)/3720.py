class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        def canReachZero(maxWeight):
            # Filter edges by weight
            filteredEdges = [edge for edge in edges if edge[2] <= maxWeight]
            
            # Group edges by source node
            outgoing = [[] for _ in range(n)]
            for a, b, w in filteredEdges:
                outgoing[a].append((b, w))
            
            # For each node, keep only the threshold smallest weight edges
            graph = [[] for _ in range(n)]
            for i in range(n):
                outgoing[i].sort(key=lambda x: x[1])  # Sort by weight
                for j in range(min(threshold, len(outgoing[i]))):
                    graph[i].append(outgoing[i][j][0])
            
            # Check if all nodes can reach node 0
            def canReach(start):
                visited = set()
                queue = [start]
                visited.add(start)
                while queue:
                    node = queue.pop(0)
                    if node == 0:
                        return True
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                return False
            
            for i in range(1, n):
                if not canReach(i):
                    return False
            return True
        
        if not edges:
            return -1
        
        weights = [edge[2] for edge in edges]
        left, right = min(weights), max(weights)
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canReachZero(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result