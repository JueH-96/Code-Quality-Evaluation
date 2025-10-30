class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        # Apply Operation 2 first where it is beneficial
        for i in range(n):
            if op2 > 0 and nums[i] >= k:
                nums[i] -= k
                op2 -= 1
        
        # Apply Operation 1 next where it is beneficial
        for i in range(n):
            if op1 > 0:
                nums[i] = (nums[i] + 1) // 2
                op1 -= 1
        
        return sum(nums)