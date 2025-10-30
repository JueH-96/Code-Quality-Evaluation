class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return float('-inf')
            max_ending_here = max_so_far = arr[0]
            for i in range(1, len(arr)):
                max_ending_here = max(arr[i], max_ending_here + arr[i])
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        # Start with the maximum subarray sum of the original array
        max_sum = kadane(nums)
        
        # Try removing each unique value
        for x in set(nums):
            new_array = [num for num in nums if num != x]
            if new_array:  # Ensure the array remains non-empty
                max_sum = max(max_sum, kadane(new_array))
        
        return max_sum