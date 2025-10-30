class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        from collections import defaultdict
        
        # Edge case where no special substrings are needed
        if k == 0:
            return True
        
        # Dictionary to store the first and last index of each character
        char_indices = defaultdict(lambda: [float('inf'), float('-inf')])
        
        # Fill dictionary with first and last occurrences of each character
        for i, char in enumerate(s):
            char_indices[char][0] = min(char_indices[char][0], i)
            char_indices[char][1] = max(char_indices[char][1], i)
        
        # List to store intervals that are completely unique
        intervals = []
        
        # Check for characters that are unique in their interval
        for char, (first, last) in char_indices.items():
            if all(first > char_indices[c][1] or last < char_indices[c][0] for c in char_indices if c != char):
                intervals.append((first, last))
        
        # Sort intervals by their starting point
        intervals.sort()
        
        # Greedy approach to select the maximum number of non-overlapping intervals
        count = 0
        end = -1
        
        for start, finish in intervals:
            if start > end:
                count += 1
                end = finish
        
        # Check if we can select at least k non-overlapping special substrings
        return count >= k