from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        a_len = len(a)
        b_len = len(b)
        
        beautiful_indices = []
        a_indices = []
        b_indices = []
        
        # Find all indices where substring matches 'a'
        for i in range(n - a_len + 1):
            if s[i:i + a_len] == a:
                a_indices.append(i)
        
        # Find all indices where substring matches 'b'
        for j in range(n - b_len + 1):
            if s[j:j + b_len] == b:
                b_indices.append(j)
        
        # Check for beautiful indices
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful_indices.append(i)
                    break
        
        return sorted(set(beautiful_indices))