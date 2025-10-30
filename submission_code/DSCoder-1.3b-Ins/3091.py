class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = (prefix[i] + nums[i]) % MOD
            suffix[i+1] = (suffix[i] + nums[n-i-1]) % MOD
        count = 0
        for i in range(n):
            if prefix[i] >= l and suffix[i+1] <= r:
                count = (count + 1) % MOD
        return count