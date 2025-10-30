from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        indices_a = [i for i in range(len(s) - len(a) + 1) if s[i:i + len(a)] == a]
        indices_b = [i for i in range(len(s) - len(b) + 1) if s[i:i + len(b)] == b]
        
        beautiful_indices = []
        
        for i in indices_a:
            left = max(0, i - k)
            right = min(len(s) - len(b), i + k)
            j = 0
            while j < len(indices_b) and indices_b[j] < left:
                j += 1
            while j < len(indices_b) and indices_b[j] <= right:
                beautiful_indices.append(i)
                break
            j += 1
        
        return beautiful_indices