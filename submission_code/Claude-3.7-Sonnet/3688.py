class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def maxSubarraySumKadane(arr):
            if not arr:  # If array is empty
                return float('-inf')  # Return negative infinity
            
            max_global = max_current = arr[0]
            for i in range(1, len(arr)):
                max_current = max(arr[i], max_current + arr[i])
                max_global = max(max_global, max_current)
            
            return max_global
        
        # First, compute maximum subarray sum for original array
        max_sum = maxSubarraySumKadane(nums)
        
        # Consider all possible values of x to remove
        distinct_elements = set(nums)
        
        for x in distinct_elements:
            # Create new array by removing all occurrences of x
            nums_without_x = [num for num in nums if num != x]
            
            # Check if removing all occurrences of x keeps the array non-empty
            if nums_without_x:
                max_sum = max(max_sum, maxSubarraySumKadane(nums_without_x))
        
        return max_sum