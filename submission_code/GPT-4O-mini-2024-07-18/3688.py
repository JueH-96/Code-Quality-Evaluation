class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            max_sum = current_sum = arr[0]
            for num in arr[1:]:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        # Calculate the maximum subarray sum without any removal
        max_sum_no_removal = kadane(nums)
        
        # To store the maximum subarray sum after removing occurrences of each unique element
        max_sum_with_removal = float('-inf')
        
        # Get unique elements to consider for removal
        unique_elements = set(nums)
        
        for x in unique_elements:
            # Create a new array without the occurrences of x
            filtered_nums = [num for num in nums if num != x]
            if filtered_nums:  # Ensure the array is not empty
                max_sum_with_removal = max(max_sum_with_removal, kadane(filtered_nums))
        
        # The result is the maximum of both scenarios
        return max(max_sum_no_removal, max_sum_with_removal)