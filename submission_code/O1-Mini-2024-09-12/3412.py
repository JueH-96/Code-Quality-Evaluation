class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_index = {c: i for i, c in enumerate(s)}
        t_index = {c: i for i, c in enumerate(t)}
        total = 0
        for c in s:
            total += abs(s_index[c] - t_index[c])
        return total