class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        inv_2 = 500000004  # Modular inverse of 2 modulo MOD
        n = len(nums)
        A = [x for x in nums if x <= k]
        
        dp = {0: 1}
        for x in A:
            temp_dp = dp.copy()
            for s in list(dp.keys()):
                new_s = s + x
                if new_s > k:
                    continue
                contribution = dp[s] * inv_2 % MOD
                if new_s in temp_dp:
                    temp_dp[new_s] = (temp_dp[new_s] + contribution) % MOD
                else:
                    temp_dp[new_s] = contribution
            dp = temp_dp
        
        S = dp.get(k, 0)
        power = pow(2, n, MOD)
        return (power * S) % MOD