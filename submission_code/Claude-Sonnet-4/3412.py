class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in t
        t_indices = {}
        for i, char in enumerate(t):
            t_indices[char] = i
        
        # Calculate the permutation difference
        total_diff = 0
        for i, char in enumerate(s):
            # Get the index of this character in t
            t_index = t_indices[char]
            # Add the absolute difference to the total
            total_diff += abs(i - t_index)
        
        return total_diff