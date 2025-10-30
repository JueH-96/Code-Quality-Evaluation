class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Since we can increase or decrease num by 1,
        # and we can do this operation at most t times,
        # the maximum achievable number would be num + t.
        return num + t