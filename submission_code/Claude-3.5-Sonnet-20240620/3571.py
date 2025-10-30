class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        coordinates.sort()
        
        # Create a mapping of y-coordinates to their indices
        y_to_index = {coord[1]: i for i, coord in enumerate(coordinates)}
        
        # Initialize dp array
        dp = [1] * n
        
        # Compute the longest increasing path for each point
        for i in range(n):
            x, y = coordinates[i]
            for j in range(i):
                prev_x, prev_y = coordinates[j]
                if prev_x < x and prev_y < y:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # Find the index of the kth coordinate in the sorted array
        k_index = next(i for i, coord in enumerate(coordinates) if coord == coordinates[k])
        
        # Initialize the result with the path ending at k
        result = dp[k_index]
        
        # Check for longer paths that include k
        x_k, y_k = coordinates[k_index]
        for i in range(k_index + 1, n):
            x, y = coordinates[i]
            if x > x_k and y > y_k:
                result = max(result, dp[i] + 1)
        
        return result