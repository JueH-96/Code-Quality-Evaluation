class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        min_max_distance = float('inf')
        n = len(points)
        for i in range(n):
            remaining_points = []
            for j in range(n):
                if i != j:
                    remaining_points.append(points[j])
            if not remaining_points:
                max_distance = 0
            else:
                max_distance = 0
                for p1_index in range(len(remaining_points)):
                    for p2_index in range(p1_index + 1, len(remaining_points)):
                        p1 = remaining_points[p1_index]
                        p2 = remaining_points[p2_index]
                        distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                        max_distance = max(max_distance, distance)
            min_max_distance = min(min_max_distance, max_distance)
        return min_max_distance