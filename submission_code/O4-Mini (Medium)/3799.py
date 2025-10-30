from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        n = len(digits)
        seen = set()
        
        # Enumerate all ways to pick three distinct positions i, j, k
        # i -> hundreds place (cannot be zero)
        # j -> tens place
        # k -> units place (must be even)
        for i in range(n):
            if digits[i] == 0:
                continue
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if digits[k] % 2 == 0:
                        num = digits[i] * 100 + digits[j] * 10 + digits[k]
                        seen.add(num)
        
        return len(seen)