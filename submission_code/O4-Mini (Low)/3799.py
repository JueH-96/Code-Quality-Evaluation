from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Count the number of distinct three-digit even numbers that can be formed
        using the given list of digits (each digit instance used at most once per number,
        no leading zeros, and the number must be even).
        """
        n = len(digits)
        seen = set()
        
        # Try all combinations of three distinct indices i, j, k
        for i in range(n):
            # Hundreds place cannot be zero
            if digits[i] == 0:
                continue
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Units place must be even
                    if digits[k] % 2 != 0:
                        continue
                    # Form the number
                    number = digits[i] * 100 + digits[j] * 10 + digits[k]
                    seen.add(number)
        
        return len(seen)