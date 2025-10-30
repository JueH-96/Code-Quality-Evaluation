class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to map each character in t to its index for O(1) lookup.
        t_index_map = {char: idx for idx, char in enumerate(t)}
        
        # Initialize the permutation difference sum.
        total_difference = 0
        
        # Iterate over the string s and calculate the absolute difference in indices.
        for idx, char in enumerate(s):
            total_difference += abs(idx - t_index_map[char])
        
        return total_difference