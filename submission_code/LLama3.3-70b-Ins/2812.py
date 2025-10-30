class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # The maximum achievable number is num + t + t, because we can decrease x by t and increase num by t
        # to make x equal to num
        return num + t + t