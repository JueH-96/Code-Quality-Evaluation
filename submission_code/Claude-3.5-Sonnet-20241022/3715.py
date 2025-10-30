class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Create a list of points with their coin values
        points = []
        for l, r, c in coins:
            points.append((l, c, 1))  # Start of segment
            points.append((r + 1, c, -1))  # End of segment + 1
        
        # Sort points by position
        points.sort()
        
        # Process points to find all unique positions and their coin values
        positions = []  # List to store (position, coins) pairs
        curr_coins = 0
        prev_pos = points[0][0]
        
        for pos, coins_val, point_type in points:
            # If position changed, add the previous position and its coin value
            if pos != prev_pos and prev_pos is not None:
                positions.append((prev_pos, curr_coins))
            
            # Update current coins based on whether we're starting or ending a segment
            curr_coins += coins_val * point_type
            prev_pos = pos
        
        # Add the last position
        positions.append((prev_pos, curr_coins))
        
        # If we have less positions than k, we need to consider gaps
        if len(positions) == 1:
            return positions[0][1] * k
        
        # Use sliding window to find maximum sum of k consecutive positions
        max_coins = 0
        curr_sum = 0
        
        # Calculate initial window sum
        for i in range(min(k, len(positions))):
            curr_sum += positions[i][1]
        max_coins = curr_sum
        
        # Slide the window
        for i in range(k, len(positions)):
            curr_sum = curr_sum - positions[i-k][1] + positions[i][1]
            max_coins = max(max_coins, curr_sum)
        
        return max_coins