class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return float('-inf')
            
            max_sum = current_sum = arr[0]
            for i in range(1, len(arr)):
                current_sum = max(arr[i], current_sum + arr[i])
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        # Get maximum subarray sum without any removal
        max_result = kadane(nums)
        
        # Try removing each unique value
        unique_values = set(nums)
        
        for val in unique_values:
            # Create array without this value
            filtered = [x for x in nums if x != val]
            
            # Only consider if array remains non-empty
            if filtered:
                max_result = max(max_result, kadane(filtered))
        
        return max_result