class Solution:
    def minimumSteps(self, s: str) -> int:
        indices = [i for i, c in enumerate(s) if c == '1']
        m = len(indices)
        n = len(s)
        if m == 0 or m == n:
            return 0
        sum_original = sum(indices)
        sum_target = m * (n - m) + m * (m - 1) // 2
        return sum_target - sum_original