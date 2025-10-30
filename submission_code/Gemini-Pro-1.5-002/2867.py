class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        ones = []
        for i in range(n):
            if nums[i] == 1:
                ones.append(i)
        
        if not ones:
            return 0
        
        ans = 1
        for i in range(1, len(ones)):
            ans = (ans * (ones[i] - ones[i-1])) % MOD
            
        return ans