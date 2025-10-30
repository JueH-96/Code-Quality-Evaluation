class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # If the number of zeros is not a multiple of 3, it's impossible
        if nums.count(0) % 3 != 0:
            return -1
        
        # Count the number of operations needed
        operations = 0
        i = 0
        while i < n - 2:
            # If we find a sequence of 3 consecutive elements that are not 1
            # we flip them
            if nums[i] != 1 or nums[i+1] != 1 or nums[i+2] != 1:
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                operations += 1
            i += 3
        
        # Check if all elements are 1
        if all(num == 1 for num in nums):
            return operations
        else:
            return -1