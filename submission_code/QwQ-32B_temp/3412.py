class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_pos = {c: i for i, c in enumerate(s)}
        t_pos = {c: i for i, c in enumerate(t)}
        total = 0
        for c in s:
            total += abs(s_pos[c] - t_pos[c])
        return total