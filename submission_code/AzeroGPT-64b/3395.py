class Solution:
    def minAnagramLength(self, s: str) -> int:
        freq = {}
        for ch in s:
            if ch in freq:
                freq[ch] += 1
            else:
                freq[ch] = 1
        
        min_length = float('inf')
        for val in freq.values():
            min_length = min(min_length, val)
        
        return min(min_length, len(freq))