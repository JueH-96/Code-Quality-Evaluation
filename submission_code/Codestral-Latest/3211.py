class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 1

        # Function to check if the array is non-decreasing
        def is_non_decreasing(arr):
            for i in range(1, len(arr)):
                if arr[i] < arr[i - 1]:
                    return False
            return True

        # If the array is already non-decreasing, return its length
        if is_non_decreasing(nums):
            return n

        # Try to find the maximum length of a non-decreasing subarray
        max_length = 1
        for i in range(n):
            for j in range(i + 1, n):
                # Create a copy of the array and replace the subarray with its sum
                temp_nums = nums[:]
                subarray_sum = sum(temp_nums[i:j + 1])
                temp_nums[i:j + 1] = [subarray_sum]

                # Check if the modified array is non-decreasing
                if is_non_decreasing(temp_nums):
                    max_length = max(max_length, len(temp_nums))

        return max_length