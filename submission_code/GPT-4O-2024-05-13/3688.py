class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        from collections import Counter
        
        def kadane(arr):
            max_ending_here = max_so_far = arr[0]
            for x in arr[1:]:
                max_ending_here = max(x, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        # Calculate the maximum subarray sum without removing any elements
        max_sum = kadane(nums)
        
        # Count the frequency of each element in the array
        freq = Counter(nums)
        
        # Try removing each unique element and calculate the maximum subarray sum
        for x in freq:
            filtered_nums = [num for num in nums if num != x]
            if filtered_nums:
                max_sum = max(max_sum, kadane(filtered_nums))
        
        return max_sum