class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Sort the coordinates based on x, then y
        sorted_coords = sorted(coordinates, key=lambda x: (x[0], x[1]))
        n = len(sorted_coords)
        
        # Create a dictionary to map each coordinate to its index in the sorted list
        coord_to_index = {tuple(coord): i for i, coord in enumerate(sorted_coords)}
        
        # Initialize dp array where dp[i] is the length of the longest path ending at sorted_coords[i]
        dp = [1] * n
        
        # The target coordinate is coordinates[k]
        target = tuple(coordinates[k])
        target_index = coord_to_index[target]
        
        # Iterate through the sorted coordinates
        for i in range(n):
            current_x, current_y = sorted_coords[i]
            # Iterate through previous coordinates to find the longest path that can be extended
            for j in range(i):
                prev_x, prev_y = sorted_coords[j]
                if prev_x < current_x and prev_y < current_y:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
        
        # The maximum path length that includes the target coordinate is dp[target_index]
        return dp[target_index]