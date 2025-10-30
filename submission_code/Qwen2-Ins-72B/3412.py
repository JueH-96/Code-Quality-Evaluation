class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        s_index = {char: i for i, char in enumerate(s)}
        t_index = {char: i for i, char in enumerate(t)}
        
        return sum(abs(s_index[char] - t_index[char]) for char in s)