class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        from collections import defaultdict

        def max_subarray_sum(arr):
            max_sum = float('-inf')
            current_sum = 0
            for num in arr:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            return max_sum

        # Calculate the maximum subarray sum without any deletions
        original_max_sum = max_subarray_sum(nums)

        # Use a dictionary to count occurrences of each element
        count = defaultdict(int)
        for num in nums:
            count[num] += 1

        # Try removing each unique element and calculate the max subarray sum
        max_sum_after_deletion = float('-inf')
        for num in count:
            # Create a new array without the current element
            new_nums = [x for x in nums if x != num]
            if new_nums:  # Ensure the array is not empty
                max_sum_after_deletion = max(max_sum_after_deletion, max_subarray_sum(new_nums))

        # Return the maximum of the original max sum and the max sum after any deletion
        return max(original_max_sum, max_sum_after_deletion)