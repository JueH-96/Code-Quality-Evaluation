from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # The smallest element while any number < k exists will always be < k,
        # so we must remove every element that is < k. Each such removal is one operation.
        return sum(num < k for num in nums)