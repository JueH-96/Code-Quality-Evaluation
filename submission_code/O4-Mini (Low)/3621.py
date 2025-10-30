from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If there are elements both below k and above k, we cannot
        # ever raise the small ones, so impossible.
        below = any(x < k for x in nums)
        above = any(x > k for x in nums)
        if below and above:
            return -1
        
        # If no element is above k:
        #   - If all are exactly k, 0 operations.
        #   - Otherwise, some are below k but none above, can't raise -> -1.
        if not above:
            return 0 if all(x == k for x in nums) else -1
        
        # Now we have no elements below k, and at least one above k.
        # We can only decrease, so k must be the final target.
        # The minimum operations is the number of distinct values > k,
        # because in each operation we pick a valid h and collapse
        # all values above h to h. To reach exactly k, we need one
        # operation per distinct value above k (in descending order).
        distinct_above = {x for x in nums if x > k}
        return len(distinct_above)