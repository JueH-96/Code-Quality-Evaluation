class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        dp = {}
        MOD = 10**9 + 7

        def solve(mask, prev):
            if mask == (1 << n) - 1:
                return 1
            
            if (mask, prev) in dp:
                return dp[(mask, prev)]
            
            ans = 0
            for i in range(n):
                if (mask >> i) & 1 == 0:
                    if prev == -1 or nums[prev] % nums[i] == 0 or nums[i] % nums[prev] == 0:
                        ans = (ans + solve(mask | (1 << i), i)) % MOD
            
            dp[(mask, prev)] = ans
            return ans

        return solve(0, -1)