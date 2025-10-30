class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can change the difference x - num by at most 2
        # (increase one and decrease the other), preserving parity.
        # To maximize x, we take the largest even shift 2*t:
        return num + 2 * t