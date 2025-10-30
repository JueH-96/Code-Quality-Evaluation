from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is less than k, we can never increase it, so impossible
        if any(x < k for x in nums):
            return -1
        # We only need one operation per distinct value strictly greater than k
        return len({x for x in nums if x > k})