from collections import defaultdict, deque

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Get unique edge weights and sort them
        weights = sorted(set(edge[2] for edge in edges))
        
        # Binary search for the minimum possible maximum weight
        left, right = 0, len(weights) - 1
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            max_weight = weights[mid]
            
            # Check if it's possible to satisfy all conditions with this max weight
            if self.is_valid(n, edges, max_weight, threshold):
                result = max_weight
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def is_valid(self, n, edges, max_weight, threshold):
        # Filter edges based on weight and organize them by source
        graph_by_source = defaultdict(list)
        for src, dest, weight in edges:
            if weight <= max_weight:
                graph_by_source[src].append((dest, weight))
        
        # Ensure each node has at most threshold outgoing edges
        # Prioritize edges with smaller weights
        final_graph = defaultdict(list)
        for src in graph_by_source:
            edges_list = sorted(graph_by_source[src], key=lambda x: x[1])
            final_graph[src] = [dest for dest, _ in edges_list[:threshold]]
        
        # Check if node 0 is reachable from all other nodes
        for node in range(1, n):
            if not self.is_reachable(node, 0, final_graph):
                return False
        
        return True
    
    def is_reachable(self, start, end, graph):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            node = queue.popleft()
            if node == end:
                return True
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False