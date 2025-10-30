import math

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        sum_all = sum(nums)

        # Helper function to find the shortest subarray in a given array `arr`
        # that sums to `target_val`.
        def get_shortest_len_for_target_in_arr(arr: List[int], target_val: int) -> float:
            if target_val == 0:
                return 0
            # Since nums[i] >= 1, sums are positive. target_val (derived from target)
            # will be non-negative in the main logic.
            if target_val < 0: 
                return math.inf

            min_len = math.inf
            current_sum = 0
            # map_val_to_idx stores {prefix_sum_value: latest_index_j}
            # where sum(arr[0...j]) = prefix_sum_value.
            # To minimize length j - idx, we need to maximize idx.
            # So, map_val_to_idx must store the largest index for any given prefix sum value.
            map_val_to_idx = {0: -1} 
            
            for j in range(len(arr)):
                current_sum += arr[j]
                
                required_prefix_sum = current_sum - target_val
                if required_prefix_sum in map_val_to_idx:
                    idx = map_val_to_idx[required_prefix_sum]
                    min_len = min(min_len, j - idx)
                
                map_val_to_idx[current_sum] = j # Always update to store the latest index
            
            return min_len

        min_total_len = math.inf
        
        # Concatenate nums with itself to handle subarrays that wrap around.
        nums_doubled = nums + nums
        
        target_mod_sum_all = target % sum_all
        
        # Iterate m = 0, 1, 2.
        # current_target_for_doubled_array = (target % sum_all) + m * sum_all
        # This is the sum we need to find in nums_doubled.
        for m in range(3):
            current_target_for_doubled_array = target_mod_sum_all + m * sum_all
            
            # If current_target_for_doubled_array is greater than target, this path is invalid
            # (unless sum_all is 0, but nums[i] >= 1 implies sum_all >= 1)
            if current_target_for_doubled_array > target : # means num_outer_cycles would be <0
                 continue

            # Number of full nums cycles accounted for outside nums_doubled.
            # (target - current_target_for_doubled_array) must be non-negative and a multiple of sum_all.
            # It will be a multiple by construction.
            num_outer_cycles = (target - current_target_for_doubled_array) // sum_all
            
            # This check is implicitly covered by `current_target_for_doubled_array > target` check above,
            # as sum_all > 0. If current_target_for_doubled_array <= target, then num_outer_cycles >= 0.
            # if num_outer_cycles < 0:
            #     continue 
            
            base_len_from_cycles = num_outer_cycles * n
            
            len_for_rem_in_doubled = get_shortest_len_for_target_in_arr(nums_doubled, current_target_for_doubled_array)
            
            if len_for_rem_in_doubled != math.inf:
                min_total_len = min(min_total_len, base_len_from_cycles + len_for_rem_in_doubled)

        if min_total_len == math.inf:
            return -1
        else:
            return int(min_total_len)