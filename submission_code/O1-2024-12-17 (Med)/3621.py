class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is less than k, we can never raise it to k (operation only decreases values),
        # so return -1 immediately.
        for val in nums:
            if val < k:
                return -1
        
        # Gather distinct values strictly greater than k.
        distinct_above_k = {val for val in nums if val > k}
        
        # The minimum number of operations equals the number of distinct "levels" above k,
        # because in each operation we can merge the largest current value down to the next largest
        # (or down to k), and that removes exactly one distinct layer.
        return len(distinct_above_k)