class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if k > max(nums):
            return -1
        
        nums.sort(reverse=True)
        operations = 0
        
        for i in range(len(nums)):
            if nums[i] <= k:
                break
            
            if i == 0 or nums[i] < nums[i-1]:
                operations += 1
        
        return operations