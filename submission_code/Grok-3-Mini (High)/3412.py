class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary to store the index of each character in t
        index_t = {char: idx for idx, char in enumerate(t)}
        
        # Initialize the sum of absolute differences
        diff_sum = 0
        
        # Iterate through each character in s with its index
        for i, char in enumerate(s):
            # Add the absolute difference of indices to the sum
            diff_sum += abs(i - index_t[char])
        
        # Return the total permutation difference
        return diff_sum