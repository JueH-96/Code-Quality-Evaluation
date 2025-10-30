class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Build a dictionary mapping each character to its index in s
        index_map_s = {char: i for i, char in enumerate(s)}
        
        # Accumulate the sum of absolute index differences for each character
        diff_sum = 0
        for i, char in enumerate(t):
            # Find the original index of char in s
            original_index = index_map_s[char]
            diff_sum += abs(original_index - i)
        
        return diff_sum