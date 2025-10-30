class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        char_index_map = {}
        
        # Map each character in s to its index
        for i, ch in enumerate(s):
            char_index_map[ch] = i
        
        # Calculate the sum of absolute index differences
        total_difference = 0
        for i, ch in enumerate(t):
            total_difference += abs(char_index_map[ch] - i)
        
        return total_difference