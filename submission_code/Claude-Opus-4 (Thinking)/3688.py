class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return float('-inf')
            max_sum = float('-inf')
            current_sum = 0
            for num in arr:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        # Maximum subarray sum for the original array
        max_result = kadane(nums)
        
        # Get unique values
        unique_values = set(nums)
        
        # Try removing each unique value
        for val in unique_values:
            # Create new array without val
            new_array = [num for num in nums if num != val]
            
            # Check if array is non-empty
            if new_array:
                max_result = max(max_result, kadane(new_array))
        
        return max_result