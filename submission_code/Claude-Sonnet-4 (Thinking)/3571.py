class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        kx, ky = coordinates[k]
        
        # Find points that can come before coordinates[k]
        before = []
        for x, y in coordinates:
            if x < kx and y < ky:
                before.append((x, y))
        
        # Find points that can come after coordinates[k]
        after = []
        for x, y in coordinates:
            if x > kx and y > ky:
                after.append((x, y))
        
        # Function to find LIS length in 2D using binary search
        def lis_2d_optimized(points):
            if not points:
                return 0
            
            # Sort by x coordinate, then by y coordinate in descending order
            points.sort(key=lambda p: (p[0], -p[1]))
            
            # Apply LIS on y coordinates
            from bisect import bisect_left
            
            tails = []
            for x, y in points:
                pos = bisect_left(tails, y)
                if pos == len(tails):
                    tails.append(y)
                else:
                    tails[pos] = y
            
            return len(tails)
        
        # Find LIS ending at coordinates[k]
        before_length = lis_2d_optimized(before)
        
        # Find LIS starting from coordinates[k]
        after_length = lis_2d_optimized(after)
        
        # Total length is before + 1 (for coordinates[k]) + after
        return before_length + 1 + after_length