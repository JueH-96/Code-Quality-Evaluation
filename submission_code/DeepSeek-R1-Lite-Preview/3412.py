class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create dictionaries to hold the index of each character in s and t
        index_s = {char: idx for idx, char in enumerate(s)}
        index_t = {char: idx for idx, char in enumerate(t)}
        
        # Initialize the total difference to 0
        total_diff = 0
        
        # Iterate through each character in s and add the absolute difference
        # between its index in s and its index in t to the total difference
        for char in s:
            total_diff += abs(index_s[char] - index_t[char])
        
        # Return the total difference
        return total_diff