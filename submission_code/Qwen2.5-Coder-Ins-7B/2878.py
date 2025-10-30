class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n < k:
            return False
        
        diff = [0] * (n + 1)
        for i in range(n):
            diff[i] += nums[i]
            if i + k < n:
                diff[i + k] -= nums[i]
        
        current = 0
        for i in range(n):
            current += diff[i]
            if current < 0:
                return False
        
        return True