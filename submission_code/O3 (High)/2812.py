class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can decrease the difference between x and num by 2
        # (move x by ±1 in one direction and num by ±1 in the opposite direction).
        # Therefore, after at most t operations, the largest achievable x satisfies:
        #     x - num = 2 * t
        return num + 2 * t