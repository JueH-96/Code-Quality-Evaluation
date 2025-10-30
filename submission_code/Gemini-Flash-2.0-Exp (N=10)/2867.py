class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        ones = []
        for i, num in enumerate(nums):
            if num == 1:
                ones.append(i)
        
        if not ones:
            return 0
        
        if len(ones) == 1:
            return 1
        
        ans = 1
        for i in range(len(ones) - 1):
            diff = ones[i+1] - ones[i]
            ans = (ans * diff) % MOD
        
        return ans