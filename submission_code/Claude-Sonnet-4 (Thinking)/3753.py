class Solution:
    def maxDifference(self, s: str) -> int:
        from collections import Counter
        
        freq = Counter(s)
        
        max_odd = 0
        min_even = float('inf')
        
        for count in freq.values():
            if count % 2 == 0:  # even frequency
                min_even = min(min_even, count)
            else:  # odd frequency
                max_odd = max(max_odd, count)
        
        return max_odd - min_even