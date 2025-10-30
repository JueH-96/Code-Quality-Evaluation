from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of pattern a
        a_indices = []
        i = 0
        while i <= len(s) - len(a):
            found = s.find(a, i)
            if found == -1:
                break
            a_indices.append(found)
            i = found + 1
        
        # Find all occurrences of pattern b
        b_indices = []
        i = 0
        while i <= len(s) - len(b):
            found = s.find(b, i)
            if found == -1:
                break
            b_indices.append(found)
            i = found + 1
        
        # Find beautiful indices using binary search
        beautiful_indices = []
        for i in a_indices:
            # Find the range of b_indices that are within distance k from i
            left = bisect_left(b_indices, i - k)
            right = bisect_right(b_indices, i + k)
            
            # If there's at least one b_index within distance k from i
            if left < right:
                beautiful_indices.append(i)
        
        return beautiful_indices