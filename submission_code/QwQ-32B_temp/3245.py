import bisect
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all starting indices for substring a in s
        A = []
        len_a = len(a)
        len_s = len(s)
        for i in range(len_s - len_a + 1):
            if s[i:i+len_a] == a:
                A.append(i)
        
        # Find all starting indices for substring b in s
        B = []
        len_b = len(b)
        for i in range(len_s - len_b + 1):
            if s[i:i+len_b] == b:
                B.append(i)
        
        result = []
        # For each index i in A, check if there's a j in B within [i-k, i+k]
        for i in A:
            low = i - k
            high = i + k
            # Find the first position in B where element >= low
            pos = bisect.bisect_left(B, low)
            # Check if there's an element at or after pos that is <= high
            if pos < len(B) and B[pos] <= high:
                result.append(i)
        
        return result