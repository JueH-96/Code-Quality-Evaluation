class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from math import inf
        
        # Prefix sums: prefix_sum[i] = sum of nums[0..i-1]
        prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        # Dictionary to store, for a given value v = nums[i],
        # the minimum prefix_sum index prefix_sum[i] among all i
        # so that we can maximize (prefix_sum[j+1] - prefix_sum[i])
        # when |nums[i] - nums[j]| == k.
        value_to_min_prefix = {}
        
        ans = -inf
        
        for i, val in enumerate(nums):
            # Check potential matches val + k and val - k
            # If k == 0, then these are the same value (val), so only check once
            if k == 0:
                potential_keys = [val]
            else:
                potential_keys = [val + k, val - k]

            for needed in potential_keys:
                if needed in value_to_min_prefix:
                    # subarray sum = prefix_sum[i+1] - (stored prefix sum)
                    candidate_sum = prefix_sum[i+1] - value_to_min_prefix[needed]
                    if candidate_sum > ans:
                        ans = candidate_sum
            
            # Update the dictionary for the current value
            current_prefix = prefix_sum[i]
            if val not in value_to_min_prefix:
                value_to_min_prefix[val] = current_prefix
            else:
                # Keep the smallest prefix sum for this value
                value_to_min_prefix[val] = min(value_to_min_prefix[val], current_prefix)
        
        # If we never updated ans, it means no good subarray was found
        return ans if ans != -inf else 0