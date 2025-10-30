from typing import List
from itertools import permutations

class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        """
        For every permutation of the three numbers:
           1. Convert each number to its binary string (without the '0b' prefix).
           2. Concatenate these strings.
           3. Interpret the concatenation as a binary number.
        The maximum value among all six permutations is returned.
        """
        best = 0                           # stores the largest value found

        for perm in permutations(nums):    # there are only 3! = 6 permutations
            # build the concatenated binary string
            bstr = ''.join(bin(x)[2:] for x in perm)
            # convert that binary string back to an integer
            val = int(bstr, 2)
            # keep the maximum
            if val > best:
                best = val

        return best