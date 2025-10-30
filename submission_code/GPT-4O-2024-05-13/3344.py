class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n = len(points)
        
        if n == 3:
            return 0
        
        max_dist = 0
        for i in range(n):
            for j in range(i + 1, n):
                max_dist = max(max_dist, manhattan_distance(points[i], points[j]))
        
        min_max_dist = float('inf')
        
        for i in range(n):
            max_dist_after_removal = 0
            for j in range(n):
                if i == j:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    max_dist_after_removal = max(max_dist_after_removal, manhattan_distance(points[j], points[k]))
            min_max_dist = min(min_max_dist, max_dist_after_removal)
        
        return min_max_dist