class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Calculate all pairwise Manhattan distances with their respective point indices
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                distances.append((dist, i, j))
        
        # Sort distances in descending order for efficiency
        distances.sort(reverse=True)
        
        min_max_distance = float('inf')
        
        # Try removing each point and find the minimum maximum distance
        for k in range(n):
            for dist, i, j in distances:
                # If this distance doesn't involve the removed point
                if i != k and j != k:
                    # This is the maximum distance after removing point k
                    # (since distances are sorted in descending order)
                    min_max_distance = min(min_max_distance, dist)
                    break
        
        return min_max_distance