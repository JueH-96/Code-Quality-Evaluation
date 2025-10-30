class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def max_distance(point_list):
            if not point_list:
                return 0
            max_dist = 0
            for i in range(len(point_list)):
                for j in range(i + 1, len(point_list)):
                    max_dist = max(max_dist, manhattan_distance(point_list[i], point_list[j]))
            return max_dist

        min_max_distance = float('inf')
        for i in range(len(points)):
            temp_points = points[:i] + points[i+1:]
            min_max_distance = min(min_max_distance, max_distance(temp_points))
        return min_max_distance