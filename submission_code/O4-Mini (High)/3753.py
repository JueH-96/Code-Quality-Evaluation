class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        freq = Counter(s)
        # Collect frequencies of characters with odd and even counts (only counts > 0 are in freq)
        odd_counts = [cnt for cnt in freq.values() if cnt % 2 == 1]
        even_counts = [cnt for cnt in freq.values() if cnt % 2 == 0]
        
        # By problem guarantee there is at least one odd and one even.
        # To maximize (odd - even), pick the largest odd and the smallest even.
        return max(odd_counts) - min(even_counts)