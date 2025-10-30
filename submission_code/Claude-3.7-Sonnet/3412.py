class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the indices of characters in t
        t_indices = {char: i for i, char in enumerate(t)}
        
        # Calculate the permutation difference
        total_diff = 0
        for i, char in enumerate(s):
            j = t_indices[char]
            total_diff += abs(i - j)
        
        return total_diff