from typing import List
import math # for infinity

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Finds the maximum sum of a subarray nums[i..j] such that |nums[i] - nums[j]| == k.

        Args:
            nums: The input list of integers.
            k: A positive integer difference.

        Returns:
            The maximum sum of a good subarray. If no good subarray exists, return 0.
        """
        # Use a dictionary to store the minimum prefix sum P[i-1] encountered so far for each value nums[i].
        # P[x] = sum(nums[0..x]), P[-1] = 0.
        # Sum(nums[i..j]) = P[j] - P[i-1].
        # We iterate j (end index) from 0 to n-1.
        # For each j, we need to find i <= j such that |nums[i] - nums[j]| == k and P[i-1] is minimized.

        min_P_before_i_for_val = {} # Maps value `v` to minimum P[idx-1] where nums[idx] == v for idx <= current j.
        current_P = 0 # Represents P[j] after processing nums[j]. Initialized to P[-1] = 0.
        max_sum = -math.inf # Initialize with negative infinity to handle negative sums
        found_good_subarray = False # Flag to track if we found at least one good subarray

        for j in range(len(nums)):
            current_num = nums[j]

            # P[j-1] is the prefix sum before index j. This is `current_P` *before* updating it.
            P_before_j = current_P

            # Update current_P to be P[j]
            current_P = P_before_j + current_num

            # We are at end index j. We look for a start index i <= j.
            # The condition is |nums[i] - nums[j]| == k, which means nums[i] = nums[j] + k or nums[i] = nums[j] - k.
            # We want to maximize sum(nums[i..j]) = P[j] - P[i-1].
            # This is equivalent to minimizing P[i-1] for a fixed P[j].

            # Target values for nums[i]
            target1 = current_num + k
            target2 = current_num - k

            # Check if we have seen target1 as nums[i] before (i <= j).
            # The dictionary stores the minimum P[idx-1] for all idx <= j where nums[idx] == target1.
            if target1 in min_P_before_i_for_val:
                # We found a potential start index i with nums[i] == target1.
                # The minimum P[i-1] for this target value is stored in the dictionary.
                min_pi_minus_1 = min_P_before_i_for_val[target1]
                potential_sum = current_P - min_pi_minus_1 # P[j] - P[i-1]
                max_sum = max(max_sum, potential_sum)
                found_good_subarray = True

            # Check if we have seen target2 as nums[i] before (i <= j).
            if target2 in min_P_before_i_for_val:
                # We found a potential start index i with nums[i] == target2.
                min_pi_minus_1 = min_P_before_i_for_val[target2]
                potential_sum = current_P - min_pi_minus_1 # P[j] - P[i-1]
                max_sum = max(max_sum, potential_sum)
                found_good_subarray = True

            # Update the dictionary with the prefix sum before index j for the current value nums[j].
            # This is `P[j-1]`, which was stored in `P_before_j`.
            # For the value `current_num = nums[j]`, the prefix sum before its index `j` is `P[j-1]`.
            # We update the minimum `P[i-1]` seen so far for the value `nums[i]`.
            # This needs to handle the case where `current_num` is seen for the first time.
            if current_num not in min_P_before_i_for_val or P_before_j < min_P_before_i_for_val[current_num]:
                 min_P_before_i_for_val[current_num] = P_before_j

        # If no good subarray was found, max_sum is still -math.inf. Return 0 in this case.
        # Otherwise, return the maximum sum found.
        if not found_good_subarray:
            return 0
        else:
            return max_sum