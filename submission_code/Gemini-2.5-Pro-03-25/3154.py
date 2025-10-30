import math
from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        """
        Calculates the maximum value of (nums[i] - nums[j]) * nums[k] for i < j < k.

        The value of a triplet (i, j, k) is (nums[i] - nums[j]) * nums[k].
        We need to find the maximum such value over all valid triplets.
        If all triplet values are negative, return 0.

        Constraints:
        3 <= nums.length <= 100
        1 <= nums[i] <= 10^6

        We can solve this efficiently in O(N) time and O(1) space using a single pass.
        Let's iterate through the array, keeping track of the necessary maximums seen so far.
        For each element `nums[k]` (at index `k`), we want to maximize `(nums[i] - nums[j]) * nums[k]`
        where `i < j < k`. Since `nums[k]` is always positive (>= 1), we need to maximize
        the term `nums[i] - nums[j]` for `i < j < k`.

        Let's iterate through the array using index `j` (from 1 to n-1). At each step `j`, we maintain:
        1. `max_num_before_j`: The maximum value of `nums[p]` for `p < j`. This is the best candidate for `nums[i]`.
        2. `max_diff_so_far`: The maximum value of `nums[p] - nums[i]` for `p < i < j`. This represents the best
           `(nums[i] - nums[j])` part found so far for triplets ending at index `k >= j`.

        Algorithm:
        Initialize `max_triplet_value = 0`.
        Initialize `max_num_before_j = nums[0]`.
        Initialize `max_diff_so_far = 0`.

        Iterate `j` from 1 to `n-1`:
            Let `current_num = nums[j]`.

            a. Consider `current_num` as `nums[k]`:
               If `j >= 2` (we need at least 3 elements for a triplet i, j, k),
               calculate the potential triplet value using the best difference found before index `j`:
               `potential_value = max_diff_so_far * current_num`.
               Update `max_triplet_value = max(max_triplet_value, potential_value)` if `max_diff_so_far > 0`.

            b. Update state for future calculations (where `current_num` might act as `nums[j]`):
               Update `max_diff_so_far`: Calculate the difference using the best `nums[i]` candidate found so far
               (`max_num_before_j`) and the current number (`current_num` acting as `nums[j]`).
               `current_diff = max_num_before_j - current_num`
               `max_diff_so_far = max(max_diff_so_far, current_diff)`

            c. Update `max_num_before_j`: Include `current_num` to potentially be the maximum for future iterations.
               `max_num_before_j = max(max_num_before_j, current_num)`

        Return `max_triplet_value`.

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The maximum triplet value, or 0 if all triplet values are negative.
        """
        n = len(nums)
        # Need at least 3 elements to form a triplet (i < j < k)
        if n < 3:
            return 0

        # Initialize the overall maximum triplet value found.
        # According to the problem statement, if all triplet values are negative, return 0.
        # So, we initialize max_triplet_value to 0.
        max_triplet_value = 0
        
        # Tracks the maximum value of nums[p] encountered for indices p < j.
        # We need this to calculate the potential difference (nums[i] - nums[j]).
        # Initialize with the first element, as it's the max for p < 1.
        # We use float('-inf') initially in case the first element is small,
        # but since nums[i] >= 1, nums[0] is sufficient.
        max_num_before_j = nums[0] 
        
        # Tracks the maximum difference (nums[p] - nums[i]) encountered for p < i < j.
        # This represents the best possible first part of the triplet calculation 
        # (nums[i] - nums[j]) using elements seen before the potential nums[k].
        # Initialize to 0, as negative differences multiplied by positive nums[k] 
        # won't yield a result greater than the initial 0.
        max_diff_so_far = 0 

        # Iterate through the array starting from the second element (index 1).
        # In each iteration 'current_idx', nums[current_idx] is processed.
        # We update the state based on considering nums[current_idx] as either
        # the middle element (nums[j]) or the last element (nums[k]) of a triplet.
        for current_idx in range(1, n):
            current_num = nums[current_idx]

            # --- Step 1: Consider current_num (nums[current_idx]) as nums[k] ---
            # We check if using the best difference found so far (max_diff_so_far, which represents
            # max(nums[p] - nums[i]) for p < i < current_idx) multiplied by the current number 
            # yields a new maximum triplet value.
            # This requires that we have processed at least indices 0 and 1 before current_idx.
            # i.e. current_idx must be at least 2.
            if current_idx >= 2:
                # Only proceed if max_diff_so_far is positive, ensuring a potential triplet value > 0.
                # Since nums[k] (current_num) >= 1.
                if max_diff_so_far > 0:
                    potential_triplet_value = max_diff_so_far * current_num
                    # Cast to int just in case of intermediate float types if we used float('-inf')
                    # but with integer inputs and max(0,...), results should remain int.
                    max_triplet_value = max(max_triplet_value, potential_triplet_value)

            # --- Step 2: Update state for future calculations ---
            # Now, prepare for future iterations where current_num might act as nums[j].
            
            # Update max_diff_so_far: 
            # Calculate the potential difference if current_num were nums[j]. The best nums[i]
            # before it would be max_num_before_j (max(nums[p] for p < current_idx)).
            # We need max_num_before_j to be valid (i.e. based on at least one element).
            # This is guaranteed as loop starts from current_idx=1 and max_num_before_j is initialized with nums[0].
            current_diff = max_num_before_j - current_num
            # Update the overall maximum difference seen so far.
            max_diff_so_far = max(max_diff_so_far, current_diff)
            
            # Update max_num_before_j: 
            # Include current_num in the calculation of the maximum element seen so far.
            # This updated value will be used in the next iteration's Step 2.
            max_num_before_j = max(max_num_before_j, current_num)
            
        # Return the maximum triplet value found. If no positive triplet was found, it remains 0.
        return int(max_triplet_value) # Ensure return type is int