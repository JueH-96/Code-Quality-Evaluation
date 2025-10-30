class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can change the difference (x - num) by -2, 0, or +2.
        # In at most t moves, the maximum achievable increase in (x - num) is 2*t.
        # Hence the largest starting x that can be brought to num in t moves is num + 2*t.
        return num + 2 * t