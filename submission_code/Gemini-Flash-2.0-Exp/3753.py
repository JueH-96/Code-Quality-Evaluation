class Solution:
    def maxDifference(self, s: str) -> int:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        max_diff = float('-inf')
        
        for char1 in counts:
            for char2 in counts:
                if (counts[char1] % 2 != 0 and counts[char2] % 2 == 0):
                    max_diff = max(max_diff, counts[char1] - counts[char2])
        
        return max_diff