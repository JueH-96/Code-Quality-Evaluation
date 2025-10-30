class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # The maximum achievable number x can be found by considering the maximum value
        # we can reach by increasing both x and num simultaneously t times.
        # Since we can increase or decrease both x and num by 1 in each operation,
        # the maximum achievable x will be num + t + t = num + 2t.
        return num + 2 * t