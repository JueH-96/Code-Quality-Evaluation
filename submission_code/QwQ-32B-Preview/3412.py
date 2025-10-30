class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Create a dictionary mapping from character to index in t
        t_indices = {char: idx for idx, char in enumerate(t)}
        # Initialize difference sum
        diff_sum = 0
        # Iterate through s, compute differences
        for idx_s, char in enumerate(s):
            idx_t = t_indices[char]
            diff_sum += abs(idx_s - idx_t)
        return diff_sum