class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        from functools import lru_cache
        
        # Sort coordinates to ensure x_i < x_i + 1 and y_i < y_i + 1
        coordinates.sort()
        
        # Create a dictionary to store the index of each coordinate
        index_map = {tuple(coord): i for i, coord in enumerate(coordinates)}
        
        # Create a graph to store the edges
        graph = defaultdict(list)
        
        # Add edges to the graph
        for i in range(len(coordinates)):
            x1, y1 = coordinates[i]
            for j in range(i + 1, len(coordinates)):
                x2, y2 = coordinates[j]
                if x2 > x1 and y2 > y1:
                    graph[i].append(j)
        
        # Use memoization to store the length of the longest path starting from each coordinate
        @lru_cache(None)
        def dfs(i):
            return 1 + max((dfs(j) for j in graph[i]), default=0)
        
        # Return the length of the longest path that contains coordinates[k]
        return dfs(index_map[tuple(coordinates[k])])