class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        index_map_s = {char: idx for idx, char in enumerate(s)}
        index_map_t = {char: idx for idx, char in enumerate(t)}
        
        permutation_difference = sum(abs(index_map_s[char] - index_map_t[char]) for char in s)
        
        return permutation_difference