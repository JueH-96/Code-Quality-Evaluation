class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        t_indices = {c: i for i, c in enumerate(t)}
        total = 0
        for idx, char in enumerate(s):
            total += abs(idx - t_indices[char])
        return total