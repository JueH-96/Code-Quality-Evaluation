class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_indices = {char: idx for idx, char in enumerate(s)}
        t_indices = {char: idx for idx, char in enumerate(t)}
        total = 0
        for char in s:
            total += abs(s_indices[char] - t_indices[char])
        return total