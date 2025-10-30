class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        from collections import defaultdict, deque
        import heapq
        
        # Build the graph
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Function to prune edges using BFS
        def prune_edges():
            # Degree of each node
            degree = defaultdict(int)
            for node in graph:
                degree[node] = len(graph[node])
            
            # Queue for nodes to process which have more than k edges
            queue = deque([node for node in degree if degree[node] > k])
            
            while queue:
                node = queue.popleft()
                if degree[node] <= k:
                    continue
                
                # Sort edges by weight in ascending order to consider removing the smallest first
                edges_to_remove = sorted(graph[node], key=lambda x: x[1])
                
                # We need to remove (degree[node] - k) edges
                num_to_remove = degree[node] - k
                removed_count = 0
                
                while removed_count < num_to_remove and edges_to_remove:
                    neighbor, weight = edges_to_remove.pop(0)
                    # Remove the edge from the graph
                    graph[node].remove((neighbor, weight))
                    graph[neighbor].remove((node, weight))
                    
                    # Decrease the degree
                    degree[node] -= 1
                    degree[neighbor] -= 1
                    
                    # If neighbor also exceeds the limit, add to queue
                    if degree[neighbor] > k:
                        queue.append(neighbor)
                    
                    removed_count += 1
        
        # Prune the edges
        prune_edges()
        
        # Calculate the sum of remaining edge weights
        visited = set()
        total_weight = 0
        
        def dfs(node):
            nonlocal total_weight
            visited.add(node)
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    total_weight += weight
                    dfs(neighbor)
        
        # Start DFS from any node, here node 0 (since it's guaranteed to be connected)
        dfs(0)
        
        return total_weight

# Example usage:
# sol = Solution()
# print(sol.maximizeSumOfWeights([[0,1,4],[0,2,2],[2,3,12],[2,4,6]], 2))  # Output: 22
# print(sol.maximizeSumOfWeights([[0,1,5],[1,2,10],[0,3,15],[3,4,20],[3,5,5],[0,6,10]], 3))  # Output: 65