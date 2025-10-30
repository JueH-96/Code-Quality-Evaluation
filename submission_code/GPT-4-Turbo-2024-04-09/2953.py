class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to store counts of points by their (x, y) values
        point_count = defaultdict(int)
        for x, y in coordinates:
            point_count[(x, y)] += 1
        
        # List of unique points
        unique_points = list(point_count.keys())
        n = len(unique_points)
        
        # Count pairs
        count = 0
        for i in range(n):
            x1, y1 = unique_points[i]
            for j in range(i + 1, n):
                x2, y2 = unique_points[j]
                if (x1 ^ x2) + (y1 ^ y2) == k:
                    # Multiply the occurrences of these points
                    count += point_count[(x1, y1)] * point_count[(x2, y2)]
        
        # Handle the case where distance is zero and points are the same
        if k == 0:
            for x, y in unique_points:
                if point_count[(x, y)] > 1:
                    # Choose 2 out of count[(x, y)] points
                    count += point_count[(x, y)] * (point_count[(x, y)] - 1) // 2
        
        return count