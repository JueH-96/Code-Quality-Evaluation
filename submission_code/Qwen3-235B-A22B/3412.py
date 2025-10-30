class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_dict = {c: i for i, c in enumerate(s)}
        t_dict = {c: i for i, c in enumerate(t)}
        total = 0
        for c in s:
            total += abs(s_dict[c] - t_dict[c])
        return total