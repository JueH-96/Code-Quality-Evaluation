class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # First, compute the maximum subarray sum without any deletions
        def kadane(arr):
            max_current = max_global = arr[0]
            for i in range(1, len(arr)):
                max_current = max(arr[i], max_current + arr[i])
                max_global = max(max_global, max_current)
            return max_global
        
        original_max = kadane(nums)
        
        # Now, for each unique element in nums, compute the maximum subarray sum after deleting all occurrences of that element
        unique_elements = set(nums)
        max_sum = original_max
        
        for x in unique_elements:
            # Create a new list without x
            new_nums = [num for num in nums if num != x]
            if not new_nums:
                continue  # Skip if the list becomes empty
            current_max = kadane(new_nums)
            if current_max > max_sum:
                max_sum = current_max
        
        return max_sum