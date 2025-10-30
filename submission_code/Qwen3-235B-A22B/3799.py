from itertools import permutations
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        valid_numbers = set()
        for perm in permutations(digits, 3):
            hundreds, tens, units = perm
            if hundreds != 0 and units % 2 == 0:
                number = hundreds * 100 + tens * 10 + units
                valid_numbers.add(number)
        return len(valid_numbers)