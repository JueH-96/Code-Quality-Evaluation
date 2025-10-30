class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Initialize variables to keep track of the maximum subarray sum and the current subarray sum
        max_sum = float('-inf')
        current_sum = 0
        
        # Iterate through the array
        for num in nums:
            # Update the current subarray sum
            current_sum = max(num, current_sum + num)
            
            # Update the maximum subarray sum
            max_sum = max(max_sum, current_sum)
        
        # Iterate through the array again, trying to remove each element
        for x in set(nums):
            # Create a new array without the current element
            new_nums = [y for y in nums if y != x]
            
            # Calculate the maximum subarray sum of the new array
            new_max_sum = self.kadane(new_nums)
            
            # Update the maximum subarray sum
            max_sum = max(max_sum, new_max_sum)
        
        return max_sum
    
    def kadane(self, nums: List[int]) -> int:
        # Implement Kadane's algorithm to find the maximum subarray sum
        max_sum = float('-inf')
        current_sum = 0
        
        for num in nums:
            current_sum = max(num, current_sum + num)
            max_sum = max(max_sum, current_sum)
        
        return max_sum