class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def can_select_k_points(min_dist):
            n = len(points)
            # Try starting from each point
            for start in range(n):
                selected = [start]
                for i in range(n):
                    if i == start:
                        continue
                    valid = True
                    for j in selected:
                        if manhattan_distance(points[i], points[j]) < min_dist:
                            valid = False
                            break
                    if valid:
                        selected.append(i)
                        if len(selected) == k:
                            return True
            return False
        
        left, right = 1, 2 * side
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if can_select_k_points(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result