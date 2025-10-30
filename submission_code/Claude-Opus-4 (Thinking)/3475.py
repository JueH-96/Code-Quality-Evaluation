class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        # Process each position from left to right
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the group of 3 starting from position i
                nums[i] = 1 - nums[i]
                nums[i + 1] = 1 - nums[i + 1]
                nums[i + 2] = 1 - nums[i + 2]
                operations += 1
        
        # Check if the last two elements are 1
        # These positions cannot be the start of a flip
        if nums[n - 2] == 0 or nums[n - 1] == 0:
            return -1
        
        return operations