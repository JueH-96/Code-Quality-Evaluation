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
        
        # Calculate max subarray sum without removing anything
        max_without_removal = kadane(nums)
        
        # Get unique values in nums
        unique_values = set(nums)
        
        # Try removing each unique value
        result = max_without_removal
        
        for value in unique_values:
            # Create array without this value
            filtered = [num for num in nums if num != value]
            
            # Skip if removing this value makes array empty
            if not filtered:
                continue
                
            # Calculate max subarray sum for this filtered array
            max_sum = kadane(filtered)
            result = max(result, max_sum)
        
        return result