class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create mapping from character to index in t
        t_index = {}
        for i, char in enumerate(t):
            t_index[char] = i
        
        # Calculate the permutation difference
        total_diff = 0
        for i, char in enumerate(s):
            total_diff += abs(i - t_index[char])
        
        return total_diff