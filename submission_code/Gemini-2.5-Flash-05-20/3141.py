import collections
from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum_nums = sum(nums)

        # Calculate the number of full repetitions of 'nums' that can fit into 'target',
        # and the remaining sum that needs to be found from a partial subarray.
        num_full_repetitions = target // total_sum_nums
        remaining_target = target % total_sum_nums

        # Initialize the base length from the full repetitions.
        # This will be part of our final answer.
        base_length_from_full_blocks = num_full_repetitions * n

        # Determine the exact sum we need to find in the partial part (required_sum_for_search)
        # and adjust the base length if target was an exact multiple of total_sum_nums.
        if remaining_target == 0 and target > 0:
            # If target is an exact multiple of total_sum_nums (e.g., target = k * S),
            # it means target is composed entirely of full 'nums' blocks.
            # We search for a subarray that sums to S (total_sum_nums) itself.
            # The length from (k-1) full blocks will be added to the length of this 'S'-summing subarray.
            required_sum_for_search = total_sum_nums
            adjusted_base_length = (num_full_repetitions - 1) * n
        else:
            # If target is not an exact multiple, we simply need to find 'remaining_target'.
            # The 'base_length_from_full_blocks' is used as is.
            required_sum_for_search = remaining_target
            adjusted_base_length = base_length_from_full_blocks

        # We search for 'required_sum_for_search' in an extended array (nums + nums).
        # This is necessary because the shortest subarray might wrap around the original 'nums' array.
        # For example, it might start at the end of 'nums' and continue at the beginning of the next 'nums' repetition.
        # Since all nums[i] >= 1, any subarray summing to 'required_sum_for_search' (which is at most total_sum_nums)
        # will have a length at most 'n'. Therefore, searching within '2*n' elements is sufficient.
        temp_nums = nums + nums
        
        # Use a hash map to store prefix sums and their earliest indices.
        # {current_sum: earliest_index_where_this_sum_was_reached}
        # Initialize with {0: -1} to correctly handle subarrays that start from index 0 of 'temp_nums'.
        prefix_sums = {0: -1} 
        current_sum = 0
        
        # Initialize the minimum length found for 'required_sum_for_search' to infinity.
        min_len_found_for_search_target = float('inf')

        # Iterate through the extended array 'temp_nums' to find the shortest subarray.
        for i in range(len(temp_nums)):
            current_sum += temp_nums[i]

            # If 'current_sum - required_sum_for_search' is in 'prefix_sums', it means we found
            # a subarray that sums to 'required_sum_for_search'.
            # The subarray starts at index (prefix_sums[current_sum - required_sum_for_search] + 1)
            # and ends at index 'i'.
            if (current_sum - required_sum_for_search) in prefix_sums:
                start_idx = prefix_sums[current_sum - required_sum_for_search]
                length = i - start_idx
                min_len_found_for_search_target = min(min_len_found_for_search_target, length)
            
            # Store the current sum and its index in the hash map.
            # Since nums[i] >= 1, 'current_sum' is strictly increasing, so we simply assign.
            # This ensures we always record the earliest index for a given sum, crucial for finding shortest length.
            prefix_sums[current_sum] = i
        
        # If 'min_len_found_for_search_target' remains infinity, it means no subarray in 'temp_nums'
        # sums to 'required_sum_for_search'. Thus, the 'target' cannot be formed.
        if min_len_found_for_search_target == float('inf'):
            return -1
        else:
            # Otherwise, the total shortest length is the sum of the adjusted base length
            # and the shortest length found for the partial/remainder sum.
            return adjusted_base_length + min_len_found_for_search_target