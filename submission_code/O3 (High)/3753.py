from collections import Counter

class Solution:
    def maxDifference(self, s: str) -> int:
        # Count frequency of each character
        freq = Counter(s)
        
        # Split counts into odd and even lists
        odd_freqs = [cnt for cnt in freq.values() if cnt % 2 == 1]
        even_freqs = [cnt for cnt in freq.values() if cnt % 2 == 0]
        
        # Per constraints, at least one odd and one even frequency exist
        max_odd  = max(odd_freqs)
        min_even = min(even_freqs)
        
        # Maximum difference = largest odd frequency minus smallest even frequency
        return max_odd - min_even