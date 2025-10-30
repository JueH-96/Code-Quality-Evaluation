import bisect
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Helper function to find all starting indices of a substring in s
        def find_all_occurrences(s, substr):
            n = len(s)
            m = len(substr)
            indices = []
            for i in range(n - m + 1):
                if s[i:i + m] == substr:
                    indices.append(i)
            return indices
        
        # Get all starting indices for a and b
        A = find_all_occurrences(s, a)
        B = find_all_occurrences(s, b)
        
        # If B is empty, no beautiful indices exist
        if not B:
            return []
        
        beautiful = []
        for i in A:
            low = i - k
            high = i + k
            # Find the first index in B >= low
            idx = bisect.bisect_left(B, low)
            # Check if the found index is within B and <= high
            if idx < len(B) and B[idx] <= high:
                beautiful.append(i)
        
        return beautiful