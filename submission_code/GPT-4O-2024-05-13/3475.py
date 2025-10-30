class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the current element and the next two elements
                nums[i] = 1 - nums[i]
                nums[i + 1] = 1 - nums[i + 1]
                nums[i + 2] = 1 - nums[i + 2]
                operations += 1
        
        # Check if the last two elements are 1
        if nums[-2] == 0 or nums[-1] == 0:
            return -1
        
        return operations