class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # The maximum achievable number x can be calculated as num + 2 * t
        # because in each operation, we can increase x by 1 and num by 1,
        # effectively increasing the difference by 2.
        return num + 2 * t