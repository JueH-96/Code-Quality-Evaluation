class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n == 3:
            return max(abs(points[0][0] - points[1][0]) + abs(points[0][1] - points[1][1]),
                       abs(points[0][0] - points[2][0]) + abs(points[0][1] - points[2][1]),
                       abs(points[1][0] - points[2][0]) + abs(points[1][1] - points[2][1]))
        
        min_distance = float('inf')
        for i in range(n):
            max_distance = 0
            for j in range(n):
                if i != j:
                    max_distance = max(max_distance, abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]))
            min_distance = min(min_distance, max_distance)
        
        return min_distance