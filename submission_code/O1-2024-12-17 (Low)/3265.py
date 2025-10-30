class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Prefix sums for quick subarray-sum calculation
        # prefix_sum[i] = sum of nums[:i]
        # subarray sum from index j..(i-1) = prefix_sum[i] - prefix_sum[j]
        prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        # Dictionary to store, for each value in nums, the minimum prefix_sum
        # seen so far at any index having that value.
        # This helps us quickly find the best (lowest) prefix_sum for the
        # "start" value that complements nums[i] to form a difference of k
        # (i.e. if nums[i] - k in dict or nums[i] + k in dict).
        min_prefix_for_value = {}
        
        answer = None  # To track if we found any valid subarray
        
        for i, val in enumerate(nums):
            # Check the two target values that would create a difference of k
            # if the subarray starts with that target and ends at val
            for target in (val + k, val - k):
                if target in min_prefix_for_value:
                    # subarray sum = prefix_sum[i+1] - prefix_of_start
                    candidate = prefix_sum[i+1] - min_prefix_for_value[target]
                    if answer is None or candidate > answer:
                        answer = candidate
            
            # Update the dictionary with the current value, using prefix_sum[i]
            # We do it after checking so we don't form a subarray of length 1
            if val not in min_prefix_for_value:
                min_prefix_for_value[val] = prefix_sum[i]
            else:
                min_prefix_for_value[val] = min(min_prefix_for_value[val], prefix_sum[i])
        
        # If we never found any subarray meeting the condition, return 0
        # Otherwise return the maximum sum found (even if negative)
        return answer if answer is not None else 0