import bisect
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of a and b in s
        A = self.find_occurrences(s, a)
        B = self.find_occurrences(s, b)
        
        # If no occurrences of b, return empty list
        if not B:
            return []
        
        res = []
        for i in A:
            left = i - k
            right = i + k
            # Find the first j in B >= left
            idx = bisect.bisect_left(B, left)
            # Check if there exists j in B such that left <= j <= right
            if idx < len(B) and B[idx] <= right:
                res.append(i)
        
        return res
    
    def find_occurrences(self, s: str, target: str) -> List[int]:
        occurrences = []
        len_t = len(target)
        n = len(s)
        if len_t == 0 or n < len_t:
            return occurrences
        
        for i in range(n - len_t + 1):
            if s[i:i + len_t] == target:
                occurrences.append(i)
        
        return occurrences