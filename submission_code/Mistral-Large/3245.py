from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        len_a = len(a)
        len_b = len(b)
        len_s = len(s)

        # Find all starting indices of substring a in s
        indices_a = [i for i in range(len_s - len_a + 1) if s[i:i + len_a] == a]

        # Find all starting indices of substring b in s
        indices_b = [i for i in range(len_s - len_b + 1) if s[i:i + len_b] == b]

        beautiful_indices = []

        # For each index of a, check if there is a valid index of b within the range
        for i in indices_a:
            for j in indices_b:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break

        return sorted(beautiful_indices)