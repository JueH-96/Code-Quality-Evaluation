import bisect
from collections import defaultdict

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        # Preprocess the target indices for each character
        char_map = defaultdict(list)
        for idx in targetIndices:
            c = source[idx]
            char_map[c].append(idx)
        
        # Sort each list for binary search
        for c in char_map:
            char_map[c].sort()
        
        count = 0
        prev = -1  # Initialize previous index to -1 to handle the first character
        
        for c in pattern:
            indices = char_map[c]
            # Find the first index in indices that is greater than prev
            i = bisect.bisect_right(indices, prev)
            if i < len(indices):
                count += 1
                prev = indices[i]
        
        return len(targetIndices) - count