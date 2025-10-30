class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in s
        s_indices = {c: i for i, c in enumerate(s)}
        
        # Initialize the permutation difference to 0
        diff = 0
        
        # Iterate through the characters in t and calculate the difference
        for i, c in enumerate(t):
            diff += abs(s_indices[c] - i)
        
        return diff