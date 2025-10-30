class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        from collections import defaultdict

        def kadane(arr):
            max_ending_here = max_so_far = arr[0]
            for x in arr[1:]:
                max_ending_here = max(x, max_ending_here + x)
                max_so_far = max(max_so_far, max_ending_here)
            return max_so_far

        # Calculate the maximum subarray sum without removing any elements
        max_sum = kadane(nums)

        # Count occurrences of each element
        count = defaultdict(int)
        for num in nums:
            count[num] += 1

        # Try removing each unique element and calculate the maximum subarray sum
        unique_nums = set(nums)
        for x in unique_nums:
            if count[x] == len(nums):
                continue  # Skip if removing x would empty the array

            # Create a new array without x
            new_arr = [num for num in nums if num != x]
            max_sum = max(max_sum, kadane(new_arr))

        return max_sum