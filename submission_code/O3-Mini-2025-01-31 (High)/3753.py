class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        freq = Counter(s)
        
        odd_frequencies = [count for count in freq.values() if count % 2 == 1]
        even_frequencies = [count for count in freq.values() if count % 2 == 0]
        
        # Maximum difference is achieved by picking the largest odd frequency 
        # and the smallest even frequency.
        return max(odd_frequencies) - min(even_frequencies)