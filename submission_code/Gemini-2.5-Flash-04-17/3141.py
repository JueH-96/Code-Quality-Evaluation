from typing import List
import math

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)

        # The target can be achieved by taking some number of full 'nums' arrays
        # and then a subarray within one or two 'nums' arrays.

        # Number of full 'nums' blocks needed
        # Use integer division
        num_blocks = target // total_sum

        # The remaining target sum needed from a partial block(s)
        remaining_target = target % total_sum

        # If the remaining target is 0, it means the original target is an exact multiple
        # of the total sum of nums. The shortest way to achieve this sum is by taking
        # exactly num_blocks copies of the nums array. This is because all elements are positive,
        # so any subarray shorter than num_blocks * n would sum to less than target.
        if remaining_target == 0:
            return num_blocks * n

        # If there's a remaining target > 0, we need to find the shortest subarray
        # within infinite_nums that sums to this remaining_target.
        # This subarray can span at most one wrap-around of the original nums array.
        # For example, it might start at the end of one 'nums' block and end at the
        # beginning of the next. Searching in nums + nums covers all such cases
        # that start within the first 'nums' block and end within the first two.
        # Since any shortest subarray in the infinite array can be shifted to start
        # within the first 'nums' block, we only need to search within nums + nums.
        extended_nums = nums + nums

        min_len_rem = float('inf')
        current_sum = 0
        l = 0 # left pointer of the sliding window

        # Use a sliding window (two pointers) to find the shortest subarray in extended_nums
        # that sums exactly to remaining_target.
        # The window is [l, r] inclusive.
        for r in range(len(extended_nums)): # right pointer of the sliding window
            current_sum += extended_nums[r]

            # While the current window sum is greater than or equal to the remaining target
            # This condition is sufficient because all nums[i] are positive, ensuring current_sum only grows or shrinks.
            while current_sum >= remaining_target:
                # If the current window sum is exactly the remaining target, update min_len_rem
                if current_sum == remaining_target:
                    min_len_rem = min(min_len_rem, r - l + 1)

                # Shrink the window from the left by moving the left pointer
                current_sum -= extended_nums[l]
                l += 1

        # If min_len_rem is still infinity, it means no subarray in extended_nums
        # sums exactly to remaining_target. Thus, the original target cannot be formed.
        if min_len_rem == float('inf'):
            return -1

        # The total minimum length is the length from the full blocks
        # plus the length of the shortest subarray for the remaining target.
        return num_blocks * n + min_len_rem