class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        total = nums[0]
        for i in range(1, len(nums)):
            total &= nums[i]
        
        if total != 0:
            return 1
        
        n = len(nums)
        suf = [0] * n
        suf[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suf[i] = nums[i] & suf[i+1]
        
        count = 0
        cur = (1 << 60) - 1
        for i in range(n):
            cur &= nums[i]
            if cur == 0:
                if i == n-1:
                    count += 1
                else:
                    if suf[i+1] == 0:
                        count += 1
                        cur = (1 << 60) - 1
        return count