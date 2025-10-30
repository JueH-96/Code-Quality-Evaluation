class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        
        nums.sort(reverse=True)
        operations = 0
        carry = 0
        
        for num in nums:
            if num + carry >= target:
                break
            
            if num > 1:
                carry += num
                operations += 1
        
        return operations