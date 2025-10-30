class Solution:
    def findPermutationDifference(self, s: str, t: str) -> int:
        """
        Calculate the permutation difference between two strings s and t.

        Args:
        s (str): original string with unique characters.
        t (str): permutation of s.

        Returns:
        int: permutation difference.
        """
        # Map each character to its index in both strings
        index_s = {ch: idx for idx, ch in enumerate(s)}
        index_t = {ch: idx for idx, ch in enumerate(t)}
        
        # Sum absolute index differences for every character
        total_diff = 0
        for ch in s:
            total_diff += abs(index_s[ch] - index_t[ch])
        
        return total_diff