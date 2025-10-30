class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n == 3:
            return 0
        
        # Calculate Manhattan distances for all pairs of points
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                distances.append((dist, i, j))
        
        # Sort distances by value
        distances.sort()
        
        # Try removing each point and find the maximum distance
        min_max_dist = float('inf')
        for i in range(n):
            max_dist = 0
            for dist, idx1, idx2 in distances:
                if idx1 != i and idx2 != i:
                    max_dist = max(max_dist, dist)
            min_max_dist = min(min_max_dist, max_dist)
        
        return min_max_dist