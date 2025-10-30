class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create dictionaries to store character indices
        s_indices = {char: i for i, char in enumerate(s)}
        t_indices = {char: i for i, char in enumerate(t)}
        
        # Calculate sum of absolute differences
        diff_sum = 0
        for char in s:
            diff_sum += abs(s_indices[char] - t_indices[char])
            
        return diff_sum