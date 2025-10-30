class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        dp = [0] * (k + 1)
        dp[0] = 1  # Base case: sum 0 with no elements
        
        for num in nums:
            temp = dp.copy()
            new_dp = [0] * (k + 1)
            for s in range(k + 1):
                if temp[s] == 0:
                    continue
                # Contribution from state1 and state2 (not in A, but can be in B)
                new_dp[s] = (new_dp[s] + temp[s] * 2) % MOD
                # Contribution from state3 (included in both B and A)
                if s + num <= k:
                    new_dp[s + num] = (new_dp[s + num] + temp[s]) % MOD
            dp = new_dp
        
        return dp[k] % MOD