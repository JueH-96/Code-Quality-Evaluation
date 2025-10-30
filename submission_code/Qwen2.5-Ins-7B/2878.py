class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        operations = [0] * (len(nums) + 1)
        
        for i in range(len(nums)):
            operations[i + 1] = operations[i] + nums[i]
            if i + k <= len(nums):
                operations[i + k] -= nums[i]
            nums[i] = operations[i + 1] - operations[i]
        
        return all(x == 0 for x in nums)