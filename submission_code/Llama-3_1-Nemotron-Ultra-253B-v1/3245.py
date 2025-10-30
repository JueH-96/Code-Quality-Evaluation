from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all starting indices for a
        a_len = len(a)
        a_positions = []
        for i in range(len(s) - a_len + 1):
            if s[i:i+a_len] == a:
                a_positions.append(i)
        
        # Find all starting indices for b
        b_len = len(b)
        b_positions = []
        for i in range(len(s) - b_len + 1):
            if s[i:i+b_len] == b:
                b_positions.append(i)
        
        # Sort the b positions to enable binary search
        b_positions.sort()
        
        result = []
        for i in a_positions:
            lower = i - k
            upper = i + k
            # Find the first position in b_positions >= lower
            pos = bisect.bisect_left(b_positions, lower)
            # Check if there's any element in the range [lower, upper]
            if pos < len(b_positions) and b_positions[pos] <= upper:
                result.append(i)
        
        return result