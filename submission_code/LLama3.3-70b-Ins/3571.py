from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Create a dictionary to store the coordinates for efficient lookups
        coord_dict = {(x, y): i for i, (x, y) in enumerate(coordinates)}
        
        # Define the directions for DFS
        directions = [(1, 1), (1, 0), (0, 1), (-1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1)]
        
        # Initialize the maximum path length
        max_length = 0
        
        # Define a helper function for DFS
        def dfs(x, y, length, visited):
            nonlocal max_length
            max_length = max(max_length, length)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in coord_dict and (nx, ny) not in visited:
                    if nx > x and ny > y:
                        dfs(nx, ny, length + 1, visited | {(nx, ny)})
        
        # Perform DFS from the given coordinate
        dfs(coordinates[k][0], coordinates[k][1], 1, {(coordinates[k][0], coordinates[k][1])})
        
        # Return the maximum path length
        return max_length