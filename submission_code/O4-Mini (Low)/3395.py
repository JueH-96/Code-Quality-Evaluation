import math
from collections import Counter

class Solution:
    def minAnagramLength(self, s: str) -> int:
        # Count frequency of each character in s
        freq = Counter(s)
        
        # Compute the gcd of all non-zero frequencies
        g = 0
        for count in freq.values():
            g = count if g == 0 else math.gcd(g, count)
        
        # The minimal length of t is total length divided by the number of blocks (g)
        return len(s) // g

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minAnagramLength("abba"))  # Output: 2
    print(sol.minAnagramLength("cdef"))  # Output: 4