class Solution:
    def minOperations(self, k: int) -> int:
        nums = [1]
        operations = 0
        
        while sum(nums) < k:
            max_num = max(nums)
            nums.append(max_num + 1)
            operations += 1
        
        return operations