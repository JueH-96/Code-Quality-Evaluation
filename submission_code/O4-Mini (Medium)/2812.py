class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # In each move we can change x-num by at most 2 (±2 or 0),
        # and parity of x-num is preserved. To reach 0 from (x-num),
        # we need |x-num| ≤ 2*t and even difference. The maximum
        # such x is num + 2*t.
        return num + 2 * t