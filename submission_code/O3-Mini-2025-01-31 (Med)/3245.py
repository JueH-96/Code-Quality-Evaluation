from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        a_len = len(a)
        b_len = len(b)
        
        # Get all starting indices where substring a occurs in s
        a_indices = []
        for i in range(n - a_len + 1):
            if s[i:i+a_len] == a:
                a_indices.append(i)
                
        # Get all starting indices where substring b occurs in s
        b_indices = []
        for j in range(n - b_len + 1):
            if s[j:j+b_len] == b:
                b_indices.append(j)
                
        # Since we want to quickly check if there is a b-index such that |i - b_index| <= k,
        # we sort b_indices (they should already be sorted, but to be safe).
        b_indices.sort()
        
        beautiful = []
        for i in a_indices:
            # Use binary search in b_indices for position where i would be inserted
            pos = bisect.bisect_left(b_indices, i)
            
            valid = False
            if pos < len(b_indices) and abs(b_indices[pos] - i) <= k:
                valid = True
            if not valid and pos > 0 and abs(b_indices[pos - 1] - i) <= k:
                valid = True
                
            if valid:
                beautiful.append(i)
        
        return beautiful