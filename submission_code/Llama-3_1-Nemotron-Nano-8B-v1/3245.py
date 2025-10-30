from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        a_len = len(a)
        b_len = len(b)
        a_indices = []
        # Collect all starting indices where a occurs in s
        for i in range(len(s) - a_len + 1):
            if s[i:i+a_len] == a:
                a_indices.append(i)
        
        # Collect all starting indices where b occurs in s
        b_indices = []
        for j in range(len(s) - b_len + 1):
            if s[j:j+b_len] == b:
                b_indices.append(j)
        
        res = []
        # For each a index, check if there's a b index within the allowed range
        for i in a_indices:
            low = i - k
            high = i + k
            # Find the first position in b_indices >= low
            pos = bisect.bisect_left(b_indices, low)
            # Check if there exists a valid j in b_indices
            if pos < len(b_indices) and b_indices[pos] <= high:
                res.append(i)
        
        return res