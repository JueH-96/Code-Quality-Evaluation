class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_indices = {char: idx for idx, char in enumerate(s)}
        t_indices = {char: idx for idx, char in enumerate(t)}
        
        return sum(abs(s_indices[char] - t_indices[char]) for char in s)