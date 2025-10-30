class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if k not in nums:
            return -1
        
        nums = sorted(nums, reverse=True)
        operations = 0
        
        for i in range(len(nums)):
            if nums[i] == k:
                break
            operations += 1
        
        return operations