from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        # Count the frequency of each character in s
        freq = Counter(s)
        
        # Separate frequencies into odd and even lists
        odd_freqs = [f for f in freq.values() if f % 2 == 1]
        even_freqs = [f for f in freq.values() if f % 2 == 0]
        
        # As per problem statement, there is at least one odd and one even frequency
        if not odd_freqs or not even_freqs:
            # Just in case, though constraints guarantee both exist
            return 0
        
        # To maximize (odd - even), pick the largest odd and the smallest even
        max_odd = max(odd_freqs)
        min_even = min(even_freqs)
        
        return max_odd - min_even