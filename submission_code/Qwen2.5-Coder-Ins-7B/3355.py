class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        total_points = sum(2 * x - 1 for x in possible)
        current_points = 0
        for i in range(len(possible) - 1):
            current_points += 2 * possible[i] - 1
            remaining_points = total_points - current_points
            if current_points > remaining_points:
                return i + 1
        return -1