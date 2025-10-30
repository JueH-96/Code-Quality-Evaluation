from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to remove each element that is strictly less than k.
        # Since each operation removes the current smallest element,
        # and we'll only remove those < k, the answer is simply
        # the count of elements in nums that are < k.
        return sum(1 for x in nums if x < k)