class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # The maximum achievable number is num + 2 * t
        # This is because for each operation, we can increase x by 1 and num by 1,
        # and we can perform this operation up to t times.
        return num + 2 * t