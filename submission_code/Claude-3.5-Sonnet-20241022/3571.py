class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        target = coordinates[k]
        
        # Create dp arrays for paths ending at and starting from each point
        dp_before = [1] * n  # Length of longest path ending at i
        dp_after = [1] * n   # Length of longest path starting from i
        
        # Sort coordinates by x and y for processing
        points = [(x, y, i) for i, (x, y) in enumerate(coordinates)]
        points.sort()
        
        # Calculate longest paths ending at each point
        for i in range(n):
            x1, y1, idx1 = points[i]
            for j in range(i):
                x2, y2, idx2 = points[j]
                if x2 < x1 and y2 < y1:
                    dp_before[idx1] = max(dp_before[idx1], dp_before[idx2] + 1)
        
        # Calculate longest paths starting from each point (reverse direction)
        points.sort(reverse=True)
        for i in range(n):
            x1, y1, idx1 = points[i]
            for j in range(i):
                x2, y2, idx2 = points[j]
                if x1 < x2 and y1 < y2:
                    dp_after[idx1] = max(dp_after[idx1], dp_after[idx2] + 1)
        
        # Return the length at target point
        return dp_before[k]