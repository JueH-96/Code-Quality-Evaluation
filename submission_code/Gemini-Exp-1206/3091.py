class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        unique_nums = sorted(counts.keys())
        
        dp = [0] * (r + 1)
        dp[0] = 1
        
        for num in unique_nums:
            new_dp = dp[:]
            for i in range(num, r + 1):
                for j in range(1, counts[num] + 1):
                    if i - j * num >= 0:
                        new_dp[i] = (new_dp[i] + dp[i - j * num]) % MOD
                    else:
                        break
            dp = new_dp
        
        ans = 0
        for i in range(l, r + 1):
            ans = (ans + dp[i]) % MOD
        
        return ans