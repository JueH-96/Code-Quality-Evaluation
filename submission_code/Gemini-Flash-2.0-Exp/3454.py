class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        
        operations = abs(diff[0])
        
        for i in range(1, n):
            operations += abs(diff[i] - diff[i-1])
            
        return operations