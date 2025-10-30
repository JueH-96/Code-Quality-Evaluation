class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        prefix = [0]*(n+1)
        suffix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = (prefix[i] + nums[i]) % MOD
        for i in range(n-1, -1, -1):
            suffix[i] = (suffix[i+1] + nums[i]) % MOD
        res = 0
        for i in range(n):
            l = i+1
            r = n
            while l < r:
                mid = (l+r+1) // 2
                if prefix[i] + suffix[mid] <= k:
                    l = mid
                else:
                    r = mid-1
            res = (res + (prefix[i] * (prefix[i]+1) // 2) % MOD * (suffix[r]-k+1) % MOD) % MOD
        return res