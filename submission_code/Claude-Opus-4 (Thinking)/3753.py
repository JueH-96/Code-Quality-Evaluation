class Solution:
    def maxDifference(self, s: str) -> int:
        # Count character frequencies
        freq = {}
        for char in s:
            freq[char] = freq.get(char, 0) + 1
        
        # Find max odd and min even frequencies
        max_odd = -1
        min_even = float('inf')
        
        for count in freq.values():
            if count % 2 == 0:
                min_even = min(min_even, count)
            else:
                max_odd = max(max_odd, count)
        
        return max_odd - min_even