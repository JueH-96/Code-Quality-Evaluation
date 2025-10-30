from itertools import permutations
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        total = 0
        even_digits = {d for d in digits if d % 2 == 0}
        
        for e in even_digits:
            if digits.count(e) == 0:
                continue
            temp = digits.copy()
            temp.remove(e)
            two_digit_set = set()
            for a, b in permutations(temp, 2):
                if a != 0:
                    two_digit_set.add(a * 10 + b)
            total += len(two_digit_set)
        
        return total