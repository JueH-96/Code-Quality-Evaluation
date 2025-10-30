from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        beautiful_indices = []
        len_s = len(s)
        len_a = len(a)
        len_b = len(b)
        
        # Find all starting indices of substring a in s
        a_indices = []
        for i in range(len_s - len_a + 1):
            if s[i:i + len_a] == a:
                a_indices.append(i)
        
        # Find all starting indices of substring b in s
        b_indices = []
        for j in range(len_s - len_b + 1):
            if s[j:j + len_b] == b:
                b_indices.append(j)
        
        # Check for each index in a_indices if there is a corresponding index in b_indices
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful_indices.append(i)
                    break  # No need to check further j's for this i
        
        return beautiful_indices