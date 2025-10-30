from typing import List
from bisect import bisect_left

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        la, lb = len(a), len(b)
        
        # Collect all starting indices where substring matches a and b
        a_indices = []
        b_indices = []
        
        for i in range(n - la + 1):
            if s[i:i+la] == a:
                a_indices.append(i)
        
        for j in range(n - lb + 1):
            if s[j:j+lb] == b:
                b_indices.append(j)
        
        # If no occurrences of b, there can be no beautiful indices
        if not b_indices:
            return []
        
        res = []
        # For each i in a_indices, check if there exists j in b_indices
        # with |j - i| <= k using binary search
        for i in a_indices:
            # We want j >= i - k
            left = i - k
            idx = bisect_left(b_indices, left)
            # Check if this j is within i + k
            if idx < len(b_indices) and b_indices[idx] <= i + k:
                res.append(i)
        
        return res