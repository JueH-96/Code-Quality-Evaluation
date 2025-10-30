class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from sortedcontainers import SortedList
        import bisect
        
        # Sort coordinates based on x first and y second
        coordinates.sort()
        
        # Extract the target point
        target_point = coordinates[k]
        
        # Create a list to store the longest increasing subsequence length ending at each point
        dp = [1] * len(coordinates)
        
        # Use a sorted list to keep track of the best y-values for each length of increasing subsequences
        # This will store tuples of (y, length)
        sorted_y_lengths = SortedList()
        
        # Process each coordinate
        for i, (x, y) in enumerate(coordinates):
            # Find the longest subsequence that can be extended by the current point
            # We need the largest length with y < current y
            idx = sorted_y_lengths.bisect_left((y, 0))  # Find the first y that is >= current y
            max_length = 0
            if idx > 0:
                max_length = sorted_y_lengths[idx - 1][1]
            
            # Update the dp value for the current point
            dp[i] = max_length + 1
            
            # Update the sorted list with the new y and length
            # We need to remove any point that has the same length but a higher y
            while sorted_y_lengths and sorted_y_lengths.bisect_left((y, dp[i])) < len(sorted_y_lengths) and sorted_y_lengths[sorted_y_lengths.bisect_left((y, dp[i]))][1] == dp[i]:
                sorted_y_lengths.pop(sorted_y_lengths.bisect_left((y, dp[i])))
            
            sorted_y_lengths.add((y, dp[i]))
        
        # The result for the point k is the value in dp at index k
        return dp[k]