from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        a_indices = [i for i in range(len(s) - len(a) + 1) if s[i:i+len(a)] == a]
        b_indices = [i for i in range(len(s) - len(b) + 1) if s[i:i+len(b)] == b]
        
        beautiful = []
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful.append(i)
                    break
        
        return beautiful