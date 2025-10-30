class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        # Create a copy of nums to manipulate
        nums_copy = nums.copy()
        for i in range(n - 2):
            if nums_copy[i] == 0:
                # Flip the current and the next two elements
                nums_copy[i] ^= 1
                nums_copy[i+1] ^= 1
                nums_copy[i+2] ^= 1
                operations += 1
        # Check if the last two elements are 1
        if nums_copy[-2] == 1 and nums_copy[-1] == 1:
            return operations
        else:
            return -1