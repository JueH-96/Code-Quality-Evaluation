import bisect
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def find_occurrences(s: str, pattern: str) -> List[int]:
            n = len(s)
            m = len(pattern)
            if m == 0 or m > n:
                return []
            occurrences = []
            for i in range(n - m + 1):
                if s[i:i+m] == pattern:
                    occurrences.append(i)
            return occurrences
        
        A = find_occurrences(s, a)
        B = find_occurrences(s, b)
        
        if not A or not B:
            return []
        
        result = []
        for i in A:
            left = bisect.bisect_left(B, i - k)
            right = bisect.bisect_right(B, i + k)
            if right > left:
                result.append(i)
        
        return result