class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can change the difference between x and num by up to 2.
        # Hence, the maximum achievable x is simply num + 2 * t.
        return num + 2 * t