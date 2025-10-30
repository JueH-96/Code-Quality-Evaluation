from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all starting indices of a in s
        indices_a = self.find_substring_indices(s, a)
        # Find all starting indices of b in s
        indices_b = self.find_substring_indices(s, b)
        # Use two-pointer approach to find beautiful indices
        result = []
        p_a = 0
        p_b = 0
        while p_a < len(indices_a) and p_b < len(indices_b):
            if indices_a[p_a] < indices_b[p_b] - k:
                p_a += 1
            elif indices_a[p_a] > indices_b[p_b] + k:
                p_b += 1
            else:
                # There exists a j such that |i - j| <= k
                result.append(indices_a[p_a])
                p_a += 1
        return result
    
    def find_substring_indices(self, s: str, sub: str) -> List[int]:
        indices = []
        start = 0
        while True:
            index = s.find(sub, start)
            if index == -1:
                break
            indices.append(index)
            start = index + 1
        return indices