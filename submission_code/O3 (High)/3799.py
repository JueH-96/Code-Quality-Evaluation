from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Count distinct three–digit even numbers that can be formed with the
        provided digits (each digit can be used at most as many times as it
        appears in the list).  No leading zeros allowed.
        """
        n = len(digits)
        seen = set()                          # stores the distinct 3–digit numbers
        
        # treat every occurrence of a digit as a separate entity by iterating
        # over indices — this naturally respects multiplicity
        for i in range(n):                    # hundreds position
            if digits[i] == 0:                # leading zero not allowed
                continue
            for j in range(n):                # tens position
                if j == i:
                    continue
                for k in range(n):            # units position
                    if k == i or k == j:
                        continue
                    if digits[k] % 2:         # must be even (units digit even)
                        continue
                    num = digits[i] * 100 + digits[j] * 10 + digits[k]
                    seen.add(num)
        
        return len(seen)