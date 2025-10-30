from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Step 1: Check for impossibility.
        # If any element in nums is less than k, it's impossible to make it k
        # because operations only decrease values (or keep them the same).
        # nums[i] > h becomes h (decrease). nums[i] <= h remains unchanged.
        if any(x < k for x in nums):
            return -1

        # Step 2: Count distinct values greater than k.
        # All elements are now guaranteed to be >= k.
        # We need to reduce elements that are > k down to k.
        # Let the distinct values in nums that are strictly greater than k be
        # d_1 > d_2 > ... > d_m > k.
        # We can perform m operations:
        # 1. Choose h = d_2 (or h = k if m=1). All d_1's become d_2. (This choice of h is valid because only d_1's are > d_2).
        #    The set of distinct values > k is now {d_2, ..., d_m}.
        # This process is repeated m times until all values > k are reduced to k.
        # Each operation reduces one "layer" of distinct values greater than k.
        # The total number of operations is m. This is minimal.
        
        # Use a set comprehension to find unique elements strictly greater than k.
        # The length of this set is the number of operations.
        distinct_elements_greater_than_k = {x for x in nums if x > k}
        
        return len(distinct_elements_greater_than_k)