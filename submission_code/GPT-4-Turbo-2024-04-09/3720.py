from typing import List
import heapq

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Create adjacency list for the graph
        from collections import defaultdict, deque
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[v].append((u, w))  # Reverse the direction for easier reachability to node 0
        
        # Function to check if all nodes can reach node 0 with max edge weight `max_weight`
        def canReachAllNodes(max_weight):
            # Create a new graph with edges only if their weight <= max_weight
            graph = defaultdict(list)
            for v in adj:
                for u, w in adj[v]:
                    if w <= max_weight:
                        graph[u].append(v)
            
            # BFS from node 0 to check reachability
            queue = deque([0])
            visited = set([0])
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == n
        
        # Binary search on the edge weight
        left, right = 1, 10**6
        result = -1
        while left <= right:
            mid = (left + right) // 2
            if canReachAllNodes(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result

# Example usage:
sol = Solution()
print(sol.minMaxWeight(5, [[1,0,1],[2,0,2],[3,0,1],[4,3,1],[2,1,1]], 2))  # Output: 1
print(sol.minMaxWeight(5, [[0,1,1],[0,2,2],[0,3,1],[0,4,1],[1,2,1],[1,4,1]], 1))  # Output: -1
print(sol.minMaxWeight(5, [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[3,4,2],[4,0,1]], 1))  # Output: 2
print(sol.minMaxWeight(5, [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[4,0,1]], 1))  # Output: -1