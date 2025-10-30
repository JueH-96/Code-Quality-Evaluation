from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        la, lb = len(a), len(b)
        
        # Collect all starting indices where b occurs in s
        b_positions = []
        for j in range(n - lb + 1):
            if s[j:j+lb] == b:
                b_positions.append(j)
        
        # b_positions is already in sorted order because we scan j increasing
        
        result = []
        # For each position i where a occurs
        for i in range(n - la + 1):
            if s[i:i+la] != a:
                continue
            
            # We need to check if there is some b_position j with |j - i| <= k
            # Use binary search on b_positions
            idx = bisect.bisect_left(b_positions, i)
            found = False
            
            # Check the candidate at idx, if any
            if idx < len(b_positions) and abs(b_positions[idx] - i) <= k:
                found = True
            # Check the one before idx, if any
            elif idx > 0 and abs(b_positions[idx-1] - i) <= k:
                found = True
            
            if found:
                result.append(i)
        
        return result