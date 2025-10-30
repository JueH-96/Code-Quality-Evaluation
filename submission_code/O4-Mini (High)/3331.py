from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # We need to remove exactly those elements that are < k,
        # each operation removes the current smallest element.
        # Therefore the minimum operations is simply the count of nums[i] < k.
        return sum(1 for num in nums if num < k)