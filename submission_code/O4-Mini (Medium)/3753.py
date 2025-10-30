from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        # Count the frequency of each character in s
        freq = Counter(s)
        
        # Separate frequencies into odd and even lists
        odd_freqs = [f for f in freq.values() if f % 2 == 1]
        even_freqs = [f for f in freq.values() if f % 2 == 0]
        
        # Since the problem guarantees at least one odd and one even,
        # we can safely compute the result.
        max_odd = max(odd_freqs)
        min_even = min(even_freqs)
        
        return max_odd - min_even