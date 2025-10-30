class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 1000000007
        n = len(nums)
        memo = {}
        
        def dp(mask, last):
            if mask == (1 << n) - 1:
                return 1
            if (mask, last) in memo:
                return memo[(mask, last)]
            ans = 0
            for next_idx in range(n):
                if (mask & (1 << next_idx)) == 0 and (nums[next_idx] % nums[last] == 0 or nums[last] % nums[next_idx] == 0):
                    ans += dp(mask | (1 << next_idx), next_idx)
                    ans %= MOD
            memo[(mask, last)] = ans
            return ans
        
        total = 0
        for start in range(n):
            total += dp(1 << start, start)
            total %= MOD
        return total