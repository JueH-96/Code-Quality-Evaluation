class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        if len(nums) != n + 1:
            return False
        
        count = [0] * (n + 1)
        for num in nums:
            count[num] += 1
        
        if count[n] != 2:
            return False
        
        for i in range(1, n):
            if count[i] != 1:
                return False
        
        return True