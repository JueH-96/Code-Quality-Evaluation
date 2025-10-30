class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)    
        if sum(nums) < n - sum(nums): return -1
        
        def f(nums, target):
            cnt = 0
            for i in range(2, n):
                if nums[i] != target or nums[i - 1] != target or nums[i - 2] != target:
                    cnt += 1
                    nums[i], nums[i - 1], nums[i - 2] = target, target, target
            return cnt
        
        return min(f(nums,1), f([1 - x for x in nums],0))