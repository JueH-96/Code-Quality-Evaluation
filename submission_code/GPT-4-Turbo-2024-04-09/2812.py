class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # The maximum achievable number x can be found by adding t to num
        # because we can decrease x by t and increase num by t to make them equal.
        return num + t