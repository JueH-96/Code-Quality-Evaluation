from typing import List

class Solution:
    """
    Finds the maximum frequency of the value k after performing the operation once.
    The operation involves selecting a subarray nums[i..j] and adding an integer x
    to all elements within that subarray.
    """
    def maxFrequency(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum frequency of k after one operation.

        Args:
            nums: The input list of integers. Constraints: 1 <= nums[i] <= 50.
            k: The target integer value. Constraint: 1 <= k <= 50.

        Returns:
            The maximum possible frequency of k after performing the operation once.
        
        The core idea is to realize that the operation transforms elements within
        the chosen subarray nums[i..j]. If we want an element nums[p] (where i <= p <= j)
        to become k, we must add x = k - nums[p]. Let the original value nums[p]
        be 'target_value'. Then x = k - target_value.

        The change in the frequency of k after applying the operation on nums[i..j]
        with a specific 'target_value' (and thus fixed x) is:
        Gain = (count of 'target_value' in nums[i..j]) - (count of 'k' in nums[i..j])

        We want to find the maximum possible gain over all choices of subarray [i..j]
        and all possible target_values (from 1 to 50, based on constraints).

        For a fixed target_value, we can define a 'gain array' where
        gain[p] = +1 if nums[p] == target_value
                  -1 if nums[p] == k
                   0 otherwise.
        The problem then reduces to finding the maximum subarray sum of this gain array.
        This is a classic problem solvable with Kadane's algorithm.

        We iterate through all possible target_values (1 to 50). For each, we run
        Kadane's algorithm on the fly (without explicitly storing the gain array)
        to find the maximum gain achievable for that target_value. The overall maximum
        gain across all target_values is tracked.

        The final maximum frequency is the initial frequency of k plus this maximum gain.
        """
        n = len(nums)
        if n == 0:
            return 0

        # Calculate the initial frequency of k in the array.
        # This serves as the baseline frequency before any operation.
        initial_k_count = 0
        for num in nums:
            if num == k:
                initial_k_count += 1

        # Initialize the maximum gain found so far across all possibilities.
        # The gain represents the net change (increase) in the frequency of k.
        # It can be 0 if no operation improves the frequency.
        max_gain = 0

        # Iterate through all possible original values (target_value) that an element
        # could have such that it becomes k after adding x. Since the element values
        # are constrained (1 to 50), the target_value must also be in this range.
        for target_value in range(1, 51):

            # Apply Kadane's algorithm to find the maximum subarray sum for the
            # effective 'gain array' corresponding to the current target_value.
            # We calculate the gain for each element on the fly.

            current_max_gain = 0       # Max gain of a subarray ending at the current position p
            global_max_gain_for_tv = 0 # Overall max gain found for this target_value across all subarrays

            for p in range(n):
                # Calculate the gain contribution of the current element nums[p]
                # if it were included in the subarray being considered for this target_value.
                gain_p = 0
                if nums[p] == target_value:
                    # If nums[p] matches the target_value, it will become k after adding x.
                    # This contributes +1 to the frequency of k.
                    gain_p += 1
                if nums[p] == k:
                    # If nums[p] was originally k, it will change to k + x after the operation.
                    # Unless x is 0 (i.e., target_value == k), this element will no longer be k.
                    # This contributes -1 to the frequency of k.
                    # Note: If target_value == k, then x=0. An element that is k (and also target_value)
                    # contributes gain_p = (+1) + (-1) = 0, correctly reflecting no change.
                    gain_p -= 1

                # Kadane's algorithm update step:
                # Calculate the maximum gain of a subarray ending at index p.
                # If the sum becomes negative, it's better to start a new subarray from the next element,
                # effectively resetting the current sum to 0. This version finds the maximum subarray sum,
                # allowing 0 if all possible subarray sums are negative or zero.
                current_max_gain = max(0, current_max_gain + gain_p)

                # Update the overall maximum gain found so far for this specific target_value.
                global_max_gain_for_tv = max(global_max_gain_for_tv, current_max_gain)

            # After processing all elements for the current target_value,
            # update the overall maximum gain found across all possible target_values.
            max_gain = max(max_gain, global_max_gain_for_tv)

        # The final maximum frequency is the initial frequency plus the maximum possible gain achieved
        # by the single allowed operation.
        return initial_k_count + max_gain