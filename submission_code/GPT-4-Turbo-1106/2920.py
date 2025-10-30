class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        # If all elements are the same, no operation is needed.
        if len(set(nums)) == 1:
            return 0
        
        n = len(nums)
        max_equal_subarray = 1
        current_subarray_length = 1
        
        # Check for the longest subarray with equal elements.
        for i in range(1, n):
            if nums[i] == nums[i - 1]:
                current_subarray_length += 1
            else:
                max_equal_subarray = max(max_equal_subarray, current_subarray_length)
                current_subarray_length = 1
        # Check for the subarray that wraps around the end to the beginning.
        if nums[0] == nums[-1]:
            i = 0
            while i < n and nums[i] == nums[-1]:
                current_subarray_length += 1
                i += 1
            max_equal_subarray = max(max_equal_subarray, current_subarray_length)
        
        # The minimum number of seconds needed is the difference between
        # the length of the array and the length of the longest subarray with equal elements.
        return n - max_equal_subarray