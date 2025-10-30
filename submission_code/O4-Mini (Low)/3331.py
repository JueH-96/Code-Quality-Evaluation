from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Each operation removes one element that is < k.
        # We need to remove all elements smaller than k.
        return sum(1 for x in nums if x < k)