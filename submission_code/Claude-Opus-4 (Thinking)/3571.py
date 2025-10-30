class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        target = coordinates[k]
        
        # Find all points that can come before target
        before = []
        for x, y in coordinates:
            if x < target[0] and y < target[1]:
                before.append((x, y))
        
        # Find all points that can come after target
        after = []
        for x, y in coordinates:
            if x > target[0] and y > target[1]:
                after.append((x, y))
        
        # Helper function to find LIS in 2D
        def lis_2d(points):
            if not points:
                return 0
            
            # Sort by x, then by y in decreasing order for same x
            points.sort(key=lambda p: (p[0], -p[1]))
            
            # Find LIS based on y-coordinates using binary search
            dp = []
            for x, y in points:
                # Binary search for the position to insert y
                left, right = 0, len(dp)
                while left < right:
                    mid = (left + right) // 2
                    if dp[mid] < y:
                        left = mid + 1
                    else:
                        right = mid
                
                if left == len(dp):
                    dp.append(y)
                else:
                    dp[left] = y
            
            return len(dp)
        
        # Find LIS ending at target
        lis_before = lis_2d(before)
        
        # Find LIS starting from target
        lis_after = lis_2d(after)
        
        # Total path length (including the target point)
        return lis_before + 1 + lis_after