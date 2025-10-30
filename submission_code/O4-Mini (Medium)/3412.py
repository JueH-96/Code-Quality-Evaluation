class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        # Build a mapping from character to its index in t
        pos_t = {ch: i for i, ch in enumerate(t)}
        
        # Sum up the absolute differences of indices for each character in s
        result = 0
        for i, ch in enumerate(s):
            result += abs(i - pos_t[ch])
        
        return result