from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations to make all elements in nums equal to k.

        An operation consists of selecting an integer h that is valid for the current
        values in nums, and for each index i where nums[i] > h, setting nums[i] to h.
        An integer h is called valid if all values in the current array that are
        strictly greater than h are identical.

        Args:
            nums: An integer array.
            k: The target integer value.

        Returns:
            The minimum number of operations required, or -1 if impossible.
        """
        # Step 1: Check if any element is less than k. If so, it's impossible.
        # The allowed operation only reduces values (nums[i] > h becomes h, and h < nums[i]).
        # Therefore, elements already less than k can never be increased to k.
        for x in nums:
            if x < k:
                return -1
        
        # Step 2: Find the distinct values in nums that are strictly greater than k.
        # Let these distinct values initially present in the array be v_1 < v_2 < ... < v_r
        # where k < v_1.
        # To make all elements equal to k, we must reduce all occurrences of these values.
        # Consider the largest distinct value v_r initially present in the array that is > k.
        # To reduce v_r, we must perform an operation with h < v_r.
        # For h to be valid, all values currently in the array strictly greater than h must be identical.
        # Let's consider the sequence of operations that reduces the largest current value > k
        # to the next largest value >= k.
        # Suppose the distinct values currently in the array are d_1 < d_2 < ... < d_p.
        # Suppose the largest value > k is d_p (so d_p > k).
        # Let the next largest value >= k be target = max({d_i | d_i < d_p and d_i >= k} U {k}).
        # If we choose h = target, are the values strictly greater than h identical?
        # Yes, the only values strictly greater than target among the distinct values {d_1, ..., d_p} are d_p.
        # So, h=target is a valid choice.
        # Applying the operation with h=target changes all occurrences of d_p to target.
        # This reduces the largest distinct value > k to a smaller value >= k.
        # This process effectively removes d_p from the set of distinct values > k (or reduces it to k).
        # This process needs to be repeated for each distinct value initially > k, starting from the largest.
        # Each distinct value initially > k requires one such "reduction phase" where it is the largest value > k
        # that needs to be eliminated.
        # The minimum number of operations is therefore the initial count of distinct values > k.
        
        distinct_greater_than_k = set()
        for x in nums:
            if x > k:
                distinct_greater_than_k.add(x)
        
        # The count of distinct values initially greater than k directly corresponds to the
        # minimum number of operations needed following the optimal strategy.
        return len(distinct_greater_than_k)