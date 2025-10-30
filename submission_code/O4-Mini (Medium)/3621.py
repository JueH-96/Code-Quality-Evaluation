from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is already below k, we can never raise it to k.
        if any(x < k for x in nums):
            return -1
        
        # Count how many distinct values are strictly greater than k.
        # Each such distinct value will require one operation to "step down"
        # from that value to the next one (eventually reaching k).
        return len({x for x in nums if x > k})