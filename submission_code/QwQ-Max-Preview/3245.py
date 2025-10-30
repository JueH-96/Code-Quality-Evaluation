import bisect
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Generate list of indices where 'a' occurs in 's'
        list_a = []
        len_a = len(a)
        len_s = len(s)
        for i in range(len_s - len_a + 1):
            if s[i:i + len_a] == a:
                list_a.append(i)
        
        # Generate list of indices where 'b' occurs in 's'
        list_b = []
        len_b = len(b)
        for j in range(len_s - len_b + 1):
            if s[j:j + len_b] == b:
                list_b.append(j)
        
        # Check each index in list_a against list_b using binary search
        beautiful = []
        for i in list_a:
            low = i - k
            high = i + k
            # Find the first position in list_b >= low
            pos = bisect.bisect_left(list_b, low)
            # Check if there's a valid j in list_b within [low, high]
            if pos < len(list_b) and list_b[pos] <= high:
                beautiful.append(i)
        
        return beautiful