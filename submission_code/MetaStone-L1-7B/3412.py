class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        t_positions = {c: idx for idx, c in enumerate(t)}
        total = 0
        for c in s:
            s_idx = s.index(c)
            t_idx = t_positions[c]
            total += abs(s_idx - t_idx)
        return total