class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return float('-inf')
            max_ending_here = max_so_far = arr[0]
            for num in arr[1:]:
                max_ending_here = max(num, max_ending_here + num)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far
        
        # Get original array's max subarray sum
        original_max = kadane(nums)
        
        # Try removing each unique number and find max subarray sum
        unique_nums = set(nums)
        max_sum = original_max
        
        for x in unique_nums:
            # Create array without x
            filtered = []
            for num in nums:
                if num != x:
                    filtered.append(num)
            
            # Only consider if array is not empty after removal
            if filtered:
                curr_max = kadane(filtered)
                max_sum = max(max_sum, curr_max)
        
        return max_sum