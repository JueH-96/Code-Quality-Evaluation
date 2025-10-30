from typing import List
import heapq
from collections import defaultdict, deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Create a graph representation
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[v].append((u, w))  # Reverse the graph for easier reachability check
        
        # Function to check if we can reach node 0 with max weight limit
        def canReachWithMaxWeight(maxWeight):
            # Create a new graph with edges that have weight <= maxWeight
            filtered_graph = defaultdict(list)
            out_degree = [0] * n
            
            for u, v, w in edges:
                if w <= maxWeight:
                    filtered_graph[u].append(v)
                    out_degree[u] += 1
            
            # Check if any node exceeds the threshold
            if any(deg > threshold for deg in out_degree):
                return False
            
            # Perform BFS/DFS from all nodes to see if we can reach node 0
            reachable = [False] * n
            queue = deque([0])
            reachable[0] = True
            
            while queue:
                node = queue.popleft()
                for neighbor in filtered_graph[node]:
                    if not reachable[neighbor]:
                        reachable[neighbor] = True
                        queue.append(neighbor)
            
            # Check if all nodes can reach node 0
            return all(reachable[i] for i in range(n) if out_degree[i] > 0)
        
        # Binary search for the minimum possible maximum weight
        left, right = 1, 10**6
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            if canReachWithMaxWeight(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result