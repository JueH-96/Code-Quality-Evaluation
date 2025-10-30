class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Map each character in t to its index
        pos_t = {ch: idx for idx, ch in enumerate(t)}
        
        # Sum the absolute differences between positions in s and t
        total_diff = 0
        for idx_s, ch in enumerate(s):
            idx_t = pos_t[ch]
            total_diff += abs(idx_s - idx_t)
        
        return total_diff