class Solution:
    def minAnagramLength(self, s: str) -> int:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        return len(counts)