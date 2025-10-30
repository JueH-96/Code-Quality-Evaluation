class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        diff = [target[i] - nums[i] for i in range(len(nums))]
        n = len(nums)
        if n == 0:
            return 0
        b = [0] * n
        b[0] = diff[0]
        for i in range(1, n):
            b[i] = diff[i] - diff[i-1]
        
        operations = 0
        for val in b:
            if val > 0:
                operations += val
                
        return operations