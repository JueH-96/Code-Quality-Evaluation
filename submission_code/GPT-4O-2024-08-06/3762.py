class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        def canAchieve(minValue):
            # Calculate the total moves needed to achieve at least minValue in all gameScore
            total_moves = 0
            for point in points:
                if point < minValue:
                    total_moves += minValue - point
            return total_moves <= m

        left, right = 0, max(points) + m
        best_min_value = 0

        while left <= right:
            mid = (left + right) // 2
            if canAchieve(mid):
                best_min_value = mid
                left = mid + 1
            else:
                right = mid - 1

        return best_min_value