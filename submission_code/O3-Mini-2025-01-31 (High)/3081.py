from typing import List
from collections import Counter

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        # Count the frequency of each number.
        freq = Counter(nums)
        # m is the maximum frequency of any number.
        m = max(freq.values())
        
        # We can only remove a pair (a, b) if a < b.
        # In other words, if one number is extremely frequent (a majority),
        # then after pairing each occurrence of a different number with one from the majority,
        # the leftover is:
        #   leftover = 2*m - n   (if m > n/2)
        #
        # Otherwise, if no element is a majority then we can remove almost all elements,
        # leaving n % 2 elements.
        if 2 * m > n:
            return 2 * m - n
        else:
            return n % 2