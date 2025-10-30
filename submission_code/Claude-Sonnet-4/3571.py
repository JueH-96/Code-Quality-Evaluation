class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        target_x, target_y = coordinates[k]
        
        # Find points that can reach target (both x and y smaller)
        before = []
        for x, y in coordinates:
            if x < target_x and y < target_y:
                before.append([x, y])
        
        # Find points reachable from target (both x and y larger)
        after = []
        for x, y in coordinates:
            if x > target_x and y > target_y:
                after.append([x, y])
        
        # Function to find LIS length in 2D
        def lis_2d(points):
            if not points:
                return 0
            
            # Sort by x coordinate, then by y coordinate in descending order
            # This ensures when we process point i, all points before it have x <= points[i][0]
            points.sort(key=lambda p: (p[0], -p[1]))
            
            # Now we need LIS based on y coordinates
            from bisect import bisect_left
            
            # dp[i] will store the smallest ending y-coordinate of increasing subsequence of length i+1
            dp = []
            
            for x, y in points:
                # Find position to insert/replace
                pos = bisect_left(dp, y)
                if pos == len(dp):
                    dp.append(y)
                else:
                    dp[pos] = y
            
            return len(dp)
        
        # Get LIS lengths for before and after
        before_length = lis_2d(before)
        after_length = lis_2d(after)
        
        # Total length is before + 1 (target point) + after
        return before_length + 1 + after_length