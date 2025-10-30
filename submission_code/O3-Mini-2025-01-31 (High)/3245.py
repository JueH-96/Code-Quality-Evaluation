from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        len_a = len(a)
        len_b = len(b)
        
        # Collect all starting indices where substring a occurs in s
        indices_a = [i for i in range(n - len_a + 1) if s[i:i+len_a] == a]
        # Collect all starting indices where substring b occurs in s
        indices_b = [j for j in range(n - len_b + 1) if s[j:j+len_b] == b]
        
        beautiful = []
        
        # For each index in indices_a, check using binary search if there exists an index in indices_b within distance k.
        for i in indices_a:
            # Find the insertion point in indices_b for index i.
            pos = bisect.bisect_left(indices_b, i)
            found = False
            # Check the candidate at pos if available.
            if pos < len(indices_b) and abs(indices_b[pos] - i) <= k:
                found = True
            # Also check the candidate just before pos if exists.
            if pos > 0 and abs(indices_b[pos - 1] - i) <= k:
                found = True
            if found:
                beautiful.append(i)
                
        return beautiful