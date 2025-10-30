from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # min_prefix_sum_with_rem[r] stores the minimum prefix sum value encountered
        # so far among all prefix sums prefix_sum[i] where the index 'i' has
        # remainder 'r' when divided by k (i.e., i % k == r).
        # The index 'i' represents the number of elements in the prefix,
        # so prefix_sum[i] = sum(nums[0]...nums[i-1]).
        # 'i' ranges from 0 to n.
        min_prefix_sum_with_rem = [math.inf] * k
        
        # The prefix sum of an empty array (nums[0]...nums[-1]) is 0.
        # This corresponds to prefix_sum[0]. The index is 0. 0 % k = 0.
        # So, the minimum prefix sum with remainder 0 seen initially is 0.
        min_prefix_sum_with_rem[0] = 0
        
        # Initialize max_sum to a very small number, as sums can be negative.
        max_sum = float('-inf')
        
        # Initialize the current prefix sum.
        current_prefix_sum = 0
        
        # Iterate through the input array nums using index j.
        # We calculate the prefix sum ending at index j.
        # This prefix sum corresponds to prefix_sum[j+1].
        # The index associated with this prefix sum is j+1.
        for j in range(n):
            # Add the current element to the running prefix sum.
            # current_prefix_sum is now prefix_sum[j+1]
            current_prefix_sum += nums[j]
            
            # The index of the current prefix sum is j+1.
            # Calculate the remainder of this index when divided by k.
            # This remainder is the required remainder for the starting
            # prefix sum index 'i' for a subarray ending at 'j' with
            # length divisible by k. (j - i + 1) % k == 0 => (j+1) % k == i % k.
            required_rem_for_start_i = (j + 1) % k
            
            # We want to find the minimum prefix sum prefix_sum[i] (where i <= j+1)
            # such that i % k == required_rem_for_start_i.
            # This minimum value is stored in min_prefix_sum_with_rem[required_rem_for_start_i].
            
            # If we have encountered a prefix sum with the required remainder before (meaning
            # min_prefix_sum_with_rem[required_rem_for_start_i] is not still math.inf),
            # we can form a valid subarray ending at index j.
            if min_prefix_sum_with_rem[required_rem_for_start_i] != math.inf:
                 # The sum of the subarray nums[i...j] is prefix_sum[j+1] - prefix_sum[i].
                 # Here, current_prefix_sum = prefix_sum[j+1], and
                 # min_prefix_sum_with_rem[required_rem_for_start_i] is the minimum prefix_sum[i]
                 # with the correct remainder.
                 current_subarray_sum = current_prefix_sum - min_prefix_sum_with_rem[required_rem_for_start_i]
                 
                 # Update the maximum sum found so far.
                 max_sum = max(max_sum, current_subarray_sum)
            
            # After processing the potential subarrays ending at index j,
            # we update the minimum prefix sum recorded for the remainder
            # corresponding to the current prefix sum index (j+1).
            # The current prefix sum is prefix_sum[j+1], its index is j+1,
            # and its remainder is (j+1) % k.
            # We update min_prefix_sum_with_rem[(j+1) % k] if current_prefix_sum is smaller.
            min_prefix_sum_with_rem[required_rem_for_start_i] = min(min_prefix_sum_with_rem[required_rem_for_start_i], current_prefix_sum)

        # The problem guarantees 1 <= k <= nums.length, ensuring at least one
        # subarray of length k exists. Thus, max_sum will be updated from
        # float('-inf') to a valid sum.
        
        return max_sum