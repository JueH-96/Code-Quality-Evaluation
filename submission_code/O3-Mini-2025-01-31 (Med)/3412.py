class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a mapping of characters to their index in the string s
        s_index_map = {char: idx for idx, char in enumerate(s)}
        # Create a mapping of characters to their index in the string t
        t_index_map = {char: idx for idx, char in enumerate(t)}
        
        # Calculate the sum of absolute differences between the indices
        total_difference = 0
        for char in s:
            total_difference += abs(s_index_map[char] - t_index_map[char])
        
        return total_difference