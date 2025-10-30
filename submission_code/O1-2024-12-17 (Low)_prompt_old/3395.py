class Solution:
    def minAnagramLength(self, s: str) -> int:
        from math import gcd
        
        # Count frequencies of letters
        freq = [0]*26
        for ch in s:
            freq[ord(ch)-ord('a')] += 1
        
        # Compute gcd of all non-zero frequencies
        curr_gcd = 0
        for f in freq:
            if f > 0:
                curr_gcd = f if curr_gcd == 0 else gcd(curr_gcd, f)
        
        # The answer is the total length of s divided by this gcd
        return len(s) // curr_gcd