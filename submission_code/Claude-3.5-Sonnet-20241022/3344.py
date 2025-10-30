class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n = len(points)
        min_max_dist = float('inf')
        
        # Try removing each point and find max distance between remaining points
        for i in range(n):
            max_dist = 0
            # Check distances between all pairs of remaining points
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    dist = manhattan_dist(points[j], points[k])
                    max_dist = max(max_dist, dist)
            
            min_max_dist = min(min_max_dist, max_dist)
            
        return min_max_dist