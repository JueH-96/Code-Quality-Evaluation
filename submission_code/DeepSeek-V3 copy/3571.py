class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Sort the coordinates based on x and then y
        sorted_coords = sorted(coordinates, key=lambda x: (x[0], x[1]))
        n = len(sorted_coords)
        
        # Create a dictionary to map each coordinate to its index in the sorted list
        coord_to_index = {tuple(coord): i for i, coord in enumerate(sorted_coords)}
        
        # Initialize a list to store the length of the longest path ending at each coordinate
        dp = [1] * n
        
        # Iterate through each coordinate to update the dp array
        for i in range(n):
            x, y = sorted_coords[i]
            # Check all possible previous coordinates that can form an increasing path
            # Since the coordinates are sorted, we only need to check the previous ones
            for j in range(i):
                prev_x, prev_y = sorted_coords[j]
                if prev_x < x and prev_y < y:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
        
        # Find the index of the k-th coordinate in the sorted list
        target_coord = tuple(coordinates[k])
        target_index = coord_to_index[target_coord]
        
        # Return the length of the longest path that includes the target coordinate
        return dp[target_index]