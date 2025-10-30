class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums for efficient subarray sum computation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_sum = 0
        # Maps a value to its minimum prefix sum
        value_to_min_prefix = {}
        
        for j in range(n):
            current_value = nums[j]
            
            # Check if nums[j] + k or nums[j] - k exists in the hashmap
            # These would be the values that would make a good subarray with nums[j]
            for target_value in [current_value + k, current_value - k]:
                if target_value in value_to_min_prefix:
                    min_prefix = value_to_min_prefix[target_value]
                    # Calculate the subarray sum from the index with the target_value to j
                    subarray_sum = prefix_sum[j + 1] - min_prefix
                    max_sum = max(max_sum, subarray_sum)
            
            # Update the minimum prefix sum for the current value
            # We want the minimum prefix sum to maximize the subarray sum
            if current_value not in value_to_min_prefix or prefix_sum[j] < value_to_min_prefix[current_value]:
                value_to_min_prefix[current_value] = prefix_sum[j]
        
        return max_sum