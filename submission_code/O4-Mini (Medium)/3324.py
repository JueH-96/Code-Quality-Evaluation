from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        """
        We can only place at most one copy of each value into each half
        without violating the distinctness constraint. Thus each value
        with frequency f can contribute at most min(f, 2) slots across
        the two halves. If the sum of these contributions is at least
        the total length of nums, we can fill both halves completely.
        """
        freq = Counter(nums)
        total_slots = sum(min(count, 2) for count in freq.values())
        return total_slots >= len(nums)