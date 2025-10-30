from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        ans = -1000000000000000  # A large negative number to represent negative infinity
        cum_sum = 0
        val_to_min_prefix = {}
        for idx in range(len(nums)):
            num_j = nums[idx]
            target1 = num_j + k
            target2 = num_j - k
            
            # Check for target1
            if target1 in val_to_min_prefix:
                sum_val = (cum_sum + num_j) - val_to_min_prefix[target1]
                ans = max(ans, sum_val)
            
            # Check for target2
            if target2 in val_to_min_prefix:
                sum_val = (cum_sum + num_j) - val_to_min_prefix[target2]
                ans = max(ans, sum_val)
            
            # Add the current index to the map with the prefix sum up to before this index
            current_prefix = cum_sum
            if num_j not in val_to_min_prefix:
                val_to_min_prefix[num_j] = current_prefix
            else:
                val_to_min_prefix[num_j] = min(val_to_min_prefix[num_j], current_prefix)
            
            # Update the cumulative sum by adding the current element
            cum_sum += num_j
        
        # If no good subarray was found, ans remains the large negative number, so return 0
        if ans == -1000000000000000:
            return 0
        else:
            return ans