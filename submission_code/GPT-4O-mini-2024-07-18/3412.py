class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        index_s = {char: i for i, char in enumerate(s)}
        index_t = {char: i for i, char in enumerate(t)}
        
        difference = 0
        for char in s:
            difference += abs(index_s[char] - index_t[char])
        
        return difference