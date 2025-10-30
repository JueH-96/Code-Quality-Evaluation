from collections import Counter
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        freq = Counter(digits)
        total = 0
        for d1 in range(1, 10):
            for d2 in range(0, 10):
                for d3 in [0, 2, 4, 6, 8]:
                    count_needed = [0] * 10
                    count_needed[d1] += 1
                    count_needed[d2] += 1
                    count_needed[d3] += 1
                    valid = True
                    for digit in range(10):
                        if count_needed[digit] > freq.get(digit, 0):
                            valid = False
                            break
                    if valid:
                        total += 1
        return total