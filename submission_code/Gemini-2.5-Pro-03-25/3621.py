from typing import List

class Solution:
    """
    Finds the minimum number of operations to make all elements in nums equal to k.

    An operation involves selecting a valid integer h and setting all nums[i] > h to h.
    h is valid if all nums[i] > h are identical.
    """
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum operations required to make all elements in nums equal to k.

        Args:
            nums: The input integer array.
            k: The target integer value.

        Returns:
            The minimum number of operations, or -1 if impossible.
        """
        
        # Step 1: Check for impossibility.
        # The operation only allows decreasing values in the array. 
        # If any element in the initial array `nums` is strictly less than `k`, 
        # it's impossible to raise its value to `k`. In such cases, we return -1.
        possible = True
        for x in nums:
            if x < k:
                possible = False
                break
        
        if not possible:
            return -1
        
        # Step 2: Calculate the minimum number of operations.
        # If all elements in `nums` are initially greater than or equal to `k`, 
        # it is always possible to reach the state where all elements are equal to `k`.
        # We need to perform operations to reduce all values strictly greater than `k` down to `k`.
        
        # Consider the unique values in `nums` that are strictly greater than `k`.
        # Let these sorted unique values be w_1 < w_2 < ... < w_q.
        # Each operation allows us to select a valid `h`. A crucial observation is that
        # we can always perform an operation to eliminate the current largest unique value 
        # greater than `k`. 
        # Let the current maximum value in the array be `m` (where `m > k`). 
        # Let `m = w_p` be the largest unique value currently greater than `k`.
        # We can choose `h` to be the next largest unique value `w_{p-1}` (if `p > 1`), 
        # or `h = k` (if `p = 1`, meaning `m` is the smallest unique value greater than `k`).
        # This choice of `h` is always valid because all elements `x` in the current array 
        # such that `x > h` must be equal to `m` (which is `w_p`). Since they are all identical, `h` is valid.
        # Performing the operation with this `h` changes all occurrences of `m` to `h`. 
        # This effectively removes `m` (`w_p`) from the set of unique values strictly greater than `k`.
        
        # Since each operation can eliminate exactly one unique value that is strictly greater than `k`,
        # the minimum number of operations required is equal to the total count of unique values 
        # in the initial array `nums` that are strictly greater than `k`.

        # Find the set of unique values present in the input array `nums`. Using a set is efficient.
        unique_nums = set(nums)
        
        # Count how many of these unique values are strictly greater than k.
        operations_count = 0
        for unique_val in unique_nums:
            # We only care about values that need to be reduced, which are those > k.
            if unique_val > k:
                operations_count += 1
                
        # Return the calculated minimum number of operations.
        return operations_count