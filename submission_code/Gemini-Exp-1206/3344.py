class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def max_distance(arr):
            max_dist = 0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    max_dist = max(max_dist, manhattan_distance(arr[i], arr[j]))
            return max_dist

        min_max_dist = float('inf')
        for i in range(len(points)):
            temp_points = points[:i] + points[i+1:]
            min_max_dist = min(min_max_dist, max_distance(temp_points))

        return min_max_dist