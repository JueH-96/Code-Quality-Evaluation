from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        a_len = len(a)
        b_len = len(b)
        n = len(s)
        
        # Find all starting indices where substring equals a
        a_indices = []
        for i in range(n - a_len + 1):
            if s[i:i + a_len] == a:
                a_indices.append(i)
        
        # Find all starting indices where substring equals b
        b_indices = []
        for i in range(n - b_len + 1):
            if s[i:i + b_len] == b:
                b_indices.append(i)
        
        result = []
        j = 0
        m = len(b_indices)
        
        for i in a_indices:
            # Move the pointer j to the first b_index >= i - k
            while j < m and b_indices[j] < i - k:
                j += 1
            # Check if the current b_index is within the range [i - k, i + k]
            if j < m and b_indices[j] <= i + k:
                result.append(i)
        
        return result