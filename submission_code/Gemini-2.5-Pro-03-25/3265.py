import math
from typing import List

class Solution:
    """
    Finds the maximum sum of a "good" subarray using an efficient single-pass approach.
    A subarray nums[i..j] is defined as "good" if the absolute difference 
    between its first element (nums[i]) and last element (nums[j]) is exactly k, 
    i.e., |nums[i] - nums[j]| == k.
    """
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        """
        Calculates the maximum sum of a good subarray.

        The algorithm iterates through the array using index `j` as the potential 
        end of a subarray. For each `nums[j]`, it efficiently finds the best 
        starting index `i <= j` such that `|nums[i] - nums[j]| == k` and the 
        subarray sum `nums[i] + ... + nums[j]` is maximized.

        This is achieved by maintaining prefix sums and using a hash map 
        (`min_prefix_sum_map`) to store the minimum prefix sum encountered so far 
        for each number value seen in the array.

        The sum of a subarray `nums[i..j]` can be calculated as 
        `prefix_sum[j+1] - prefix_sum[i]`, where `prefix_sum[p]` is the sum of 
        elements `nums[0...p-1]` (`prefix_sum[0] = 0`). To maximize this sum for a 
        fixed `j`, we need to minimize `prefix_sum[i]`.

        The hash map `min_prefix_sum_map` stores `value -> min(prefix_sum[i])` for 
        all indices `i` processed so far where `nums[i] == value`.

        Args:
            nums: The input list of integers.
            k: The required positive absolute difference between the first and 
               last elements of a good subarray.

        Returns:
            The maximum sum found among all good subarrays. If no good subarray 
            exists, it returns 0.
        """
        n = len(nums)
        
        # Initialize max_sum to negative infinity. This handles cases where the 
        # maximum sum itself is negative. If no good subarray is found, we will return 0.
        max_sum = -math.inf 
        
        # This flag tracks whether at least one good subarray has been found.
        found_good_subarray = False
        
        # The map stores: number_value -> minimum_prefix_sum_ending_before_index_i
        # where nums[i] == number_value. Specifically, map[val] holds the minimum 
        # prefix_sum[i] (sum of nums[0...i-1]) encountered for any index i where 
        # nums[i] == val.
        min_prefix_sum_map = {}
        
        # current_prefix_sum keeps track of the prefix sum up to the element *before* 
        # the current index j. It represents prefix_sum[j] at the start of iteration j.
        current_prefix_sum = 0 

        # Iterate through the array with j as the index of the potential last element.
        for j in range(n):
            # prefix_sum[j] = sum(nums[0...j-1]) is the value of current_prefix_sum
            # before processing element j.
            prefix_sum_j = current_prefix_sum 
            current_num = nums[j]

            # Calculate prefix_sum[j+1] = sum(nums[0...j])
            prefix_sum_j_plus_1 = current_prefix_sum + nums[j]

            # --- Check if nums[j] can be the end of a good subarray ---
            # We need to find if there exists a starting index i < j such that:
            # 1. nums[i] == current_num - k 
            # 2. nums[i] == current_num + k
            # If such an i exists, we calculate the subarray sum using the minimum 
            # prefix_sum[i] stored in our map for that value of nums[i].

            # Case 1: Look for a starting element nums[i] == current_num - k
            target_start_val_1 = current_num - k
            if target_start_val_1 in min_prefix_sum_map:
                # Get the minimum prefix_sum[i] for starts where nums[i] = target_start_val_1
                min_prefix_sum_i = min_prefix_sum_map[target_start_val_1]
                # Calculate sum(nums[i..j]) = prefix_sum[j+1] - prefix_sum[i]
                current_subarray_sum = prefix_sum_j_plus_1 - min_prefix_sum_i
                # Update the overall maximum sum found so far.
                max_sum = max(max_sum, current_subarray_sum)
                found_good_subarray = True # A good subarray was found.

            # Case 2: Look for a starting element nums[i] == current_num + k
            target_start_val_2 = current_num + k
            if target_start_val_2 in min_prefix_sum_map:
                # Get the minimum prefix_sum[i] for starts where nums[i] = target_start_val_2
                min_prefix_sum_i = min_prefix_sum_map[target_start_val_2]
                # Calculate sum(nums[i..j]) = prefix_sum[j+1] - prefix_sum[i]
                current_subarray_sum = prefix_sum_j_plus_1 - min_prefix_sum_i
                # Update the overall maximum sum found so far.
                max_sum = max(max_sum, current_subarray_sum)
                found_good_subarray = True # A good subarray was found.

            # --- Update the map for future iterations ---
            # The current element nums[j] (with value current_num) could be the 
            # starting element nums[i] for a future good subarray ending at index > j.
            # We need to store its associated prefix sum, which is prefix_sum[j] 
            # (the sum *before* element nums[j]).
            # We only store/update if this prefix_sum[j] is the minimum seen so far 
            # for the value current_num.
            
            # Use map.get(key, default) to handle the first time a number is encountered.
            # math.inf ensures the first prefix sum associated with a number is stored.
            min_prefix_sum_map[current_num] = min(min_prefix_sum_map.get(current_num, math.inf), prefix_sum_j)

            # Update current_prefix_sum for the next iteration. It should hold prefix_sum[j+1].
            current_prefix_sum = prefix_sum_j_plus_1

        # After checking all possible subarrays:
        # If no good subarray was ever found, return 0 as required.
        if not found_good_subarray:
            return 0
        else:
            # Otherwise, return the maximum sum found. 
            # Since nums and k are integers, sums will be integers. max_sum will hold 
            # the integer maximum sum unless it remained -math.inf (handled by the check above).
            return max_sum