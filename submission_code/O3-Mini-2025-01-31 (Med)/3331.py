from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Since the only allowed operation is to remove the smallest element,
        # and we want all elements to be >= k, we need to remove every element less than k.
        # The operation order is forced by the problem, but essentially the number of operations
        # is exactly the count of elements in nums that are less than k.
        return sum(1 for num in nums if num < k)