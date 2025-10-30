from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Get list of indices where substring starting equals pattern.
        a_indices = []
        b_indices = []
        
        n = len(s)
        len_a = len(a)
        len_b = len(b)
        
        # Collect indices for substring a in s
        for i in range(n - len_a + 1):
            if s[i:i + len_a] == a:
                a_indices.append(i)
        
        # Collect indices for substring b in s
        for j in range(n - len_b + 1):
            if s[j:j + len_b] == b:
                b_indices.append(j)
        
        # b_indices is already sorted as found in order.
        beautiful = []
        
        # For each index in a_indices, check if there exists some index in b_indices
        # that is within the range [i - k, i + k]
        for i in a_indices:
            # lower bound for j should be i-k
            left_bound = i - k
            right_bound = i + k
            
            # Find the first index in b_indices that is >= left_bound
            pos = bisect_left(b_indices, left_bound)
            if pos < len(b_indices) and b_indices[pos] <= right_bound:
                beautiful.append(i)
        
        return beautiful