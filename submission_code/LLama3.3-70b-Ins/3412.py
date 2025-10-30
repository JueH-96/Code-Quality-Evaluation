class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in string s
        s_index = {char: i for i, char in enumerate(s)}
        
        # Initialize the permutation difference
        permutation_diff = 0
        
        # Iterate over the characters in string t
        for i, char in enumerate(t):
            # Calculate the absolute difference between the index of the occurrence of the character in s and the index of the occurrence of the same character in t
            permutation_diff += abs(s_index[char] - i)
        
        # Return the permutation difference
        return permutation_diff