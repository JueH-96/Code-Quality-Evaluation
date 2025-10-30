from typing import List
import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of a and b in s
        a_len = len(a)
        b_len = len(b)
        A = []
        B = []
        s_len = len(s)
        
        for i in range(s_len - a_len + 1):
            if s[i:i + a_len] == a:
                A.append(i)
        
        for i in range(s_len - b_len + 1):
            if s[i:i + b_len] == b:
                B.append(i)
        
        res = []
        for i in A:
            low = i - k
            high = i + k
            idx = bisect.bisect_left(B, low)
            if idx < len(B) and B[idx] <= high:
                res.append(i)
        
        return res