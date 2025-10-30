class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_pos = {char: idx for idx, char in enumerate(s)}
        t_pos = {char: idx for idx, char in enumerate(t)}
        total = 0
        for char in s:
            total += abs(s_pos[char] - t_pos[char])
        return total