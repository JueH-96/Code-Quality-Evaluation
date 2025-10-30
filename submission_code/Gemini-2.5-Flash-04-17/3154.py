from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        """
        Finds the maximum value of (nums[i] - nums[j]) * nums[k] for all triplets (i, j, k)
        such that i < j < k. Returns 0 if all such triplets have a negative value.
        Uses O(n) time and O(1) space.

        The logic involves a single pass through the array, maintaining three values:
        1. `max_i`: The maximum value encountered in the array so far (up to the current index).
        2. `max_diff`: The maximum value of `nums[i] - nums[j]` encountered so far, where `i < j`
           and `j` is an index prior to or equal to the current index.
        3. `max_val`: The maximum triplet value `(nums[i] - nums[j]) * nums[k]` found so far.

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The maximum triplet value, or 0 if all are negative.
        """
        n = len(nums)
        
        # max_val: Stores the maximum triplet value found so far.
        # Initialized to negative infinity to correctly track the maximum,
        # including cases where all triplet values are negative.
        # We will return max(0, max_val) at the end.
        max_val = float('-inf') 
        
        # max_i: Stores the maximum value of nums[i] encountered so far, considering indices < current index.
        # When we are at index `curr`, `max_i` stores `max(nums[0]...nums[curr-1])`.
        # Initialize with the first element as it's the only element before index 1.
        # Since nums[i] >= 1, initializing with 0 is also safe if we handle the first element properly.
        # Let's initialize with nums[0].
        max_i = nums[0]
        
        # max_diff: Stores the maximum value of (nums[i] - nums[j]) encountered so far, considering indices i < j <= current index - 1.
        # When we are at index `curr`, `max_diff` stores `max(nums[p] - nums[q])` for `p < q <= curr-1`.
        # This represents the maximum possible difference term for any triplet ending at or before index `curr-1`.
        # Initialize to negative infinity to correctly track the maximum difference,
        # which could be negative.
        max_diff = float('-inf')

        # Iterate through the array starting from the second element (index 1).
        # The current index `curr` can be thought of as the potential position for `j` or `k`.
        # As we iterate, we update max_i and max_diff which represent states derived from indices < curr.
        for curr in range(1, n):
            # At index `curr`, nums[curr] is the current element.

            # Step 1: Consider nums[curr] as nums[k] in a triplet (i, j, k) where i < j < k = curr.
            # The triplet value is (nums[i] - nums[j]) * nums[curr].
            # To maximize this, we need the maximum possible value of (nums[i] - nums[j]) for i < j < curr.
            # The variable `max_diff` calculated in the previous step (when `curr-1` was the current index)
            # holds exactly `max(nums[p] - nums[q])` for `p < q <= curr-1`.
            # This `max_diff` is exactly `max(nums[i] - nums[j])` for `i < j < curr`.
            # We update `max_val` with the potential maximum triplet value ending at `nums[curr]`.
            # max_val = max(max_val, max(nums[i] - nums[j] for i < j < curr) * nums[curr])
            max_val = max(max_val, max_diff * nums[curr])
            
            # Step 2: Consider nums[curr] as nums[j] in a potential difference (nums[i] - nums[j]) where i < j = curr.
            # The difference is (nums[i] - nums[curr]).
            # To maximize this difference, we need the maximum `nums[i]` where `i < curr`.
            # The variable `max_i` calculated up to the previous step (when `curr-1` was the current index)
            # holds exactly `max(nums[0]...nums[curr-1])`.
            # This `max_i` is exactly `max(nums[p])` for `p < curr`.
            # We update `max_diff` with this potential maximum difference ending at `nums[curr]` (as j).
            # `max_diff` will now store `max(nums[p] - nums[q])` for `p < q <= curr`.
            # max_diff = max(max_diff, max(nums[i] for i < curr) - nums[curr])
            max_diff = max(max_diff, max_i - nums[curr])
            
            # Step 3: Update `max_i` to include the current element `nums[curr]` for the next iteration.
            # `max_i` will now store `max(nums[0]...nums[curr])`.
            max_i = max(max_i, nums[curr])

        # According to the problem statement:
        # If all such triplets have a negative value, return 0.
        # Otherwise, return the maximum value found (which could be 0 or positive).
        # This is handled by taking the maximum of 0 and max_val.
        # If max_val is negative, max(0, max_val) is 0.
        # If max_val is non-negative (0 or positive), max(0, max_val) is max_val.
        return max(0, max_val)