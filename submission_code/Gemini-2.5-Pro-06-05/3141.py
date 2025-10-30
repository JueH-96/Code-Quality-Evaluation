import math
from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        
        n = len(nums)
        total_sum = sum(nums)
        
        def find_shortest_in_2n(t: int) -> int:
            """
            Helper function to find the shortest subarray with sum t in nums + nums.
            This covers subarrays fully within nums and those that wrap around.
            """
            if t < 0:
                return float('inf')
            
            prefix_map = {0: -1}
            current_sum = 0
            min_l = float('inf')
            
            # We search in an array of size 2*n to handle wrap-around cases.
            for i in range(2 * n):
                current_sum += nums[i % n]
                
                if current_sum - t in prefix_map:
                    min_l = min(min_l, i - prefix_map[current_sum - t])

                prefix_map[current_sum] = i
                
            return min_l

        min_len = float('inf')
        
        num_reps = target // total_sum
        rem_target = target % total_sum

        # Candidate 1: Remainder sum is rem_target
        if rem_target == 0:
            # If target is a multiple of total_sum, one solution is simply
            # `num_reps` full copies of `nums`.
            min_len = num_reps * n
        else:
            len_rem = find_shortest_in_2n(rem_target)
            if len_rem != float('inf'):
                min_len = min(min_len, num_reps * n + len_rem)

        # Candidate 2: Remainder sum is rem_target + total_sum
        # This corresponds to using num_reps - 1 full blocks.
        if num_reps >= 1:
            len_rem = find_shortest_in_2n(rem_target + total_sum)
            if len_rem != float('inf'):
                min_len = min(min_len, (num_reps - 1) * n + len_rem)

        # Candidate 3: Remainder sum is rem_target + 2*total_sum.
        # This is only possible if rem_target is 0 (so s_rem = 2*total_sum)
        # and we use `num_reps - 2` full blocks.
        if rem_target == 0 and num_reps >= 2:
            len_rem = find_shortest_in_2n(2 * total_sum)
            if len_rem != float('inf'):
                min_len = min(min_len, (num_reps - 2) * n + len_rem)

        return min_len if min_len != float('inf') else -1