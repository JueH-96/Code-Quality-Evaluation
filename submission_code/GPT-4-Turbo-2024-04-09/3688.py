class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def kadane(arr):
            max_ending_here = max_so_far = arr[0]
            for x in arr[1:]:
                max_ending_here = max(x, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        # Calculate the maximum subarray sum of the original array
        max_sum = kadane(nums)
        
        # Count frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        # For each unique number, calculate the max subarray sum if we remove all occurrences of that number
        for num in freq:
            if freq[num] < len(nums):  # We can only consider removing num if it doesn't result in an empty array
                filtered_nums = [x for x in nums if x != num]
                max_sum = max(max_sum, kadane(filtered_nums))
        
        return max_sum