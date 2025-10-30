class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # In one operation, we can reduce the difference between x and num by 2
        # (by decreasing x and increasing num if x > num, or vice versa).
        # Therefore, starting from an achievable x, the maximum difference we
        # can fix within t operations is 2*t.
        # For achieving equality, we need |x - num| <= 2*t and x must have the same parity as num.
        # The maximum such x is when x = num + 2*t.
        return num + 2 * t