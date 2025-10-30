class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        min_distance = float('inf')
        for i in range(len(points)):
            max_distance = 0
            for j in range(len(points)):
                if j != i:
                    for k in range(len(points)):
                        if k != i and k != j:
                            current_distance = manhattan(points[j], points[k])
                            max_distance = max(max_distance, current_distance)
            min_distance = min(min_distance, max_distance)
        
        return min_distance