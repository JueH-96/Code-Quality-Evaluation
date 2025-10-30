class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        max_s = 800
        # Precompute memo for x from 1 to max_s
        memo = [0] * (max_s + 1)
        memo[1] = 0
        for x in range(2, max_s + 1):
            cnt = bin(x).count('1')
            memo[x] = 1 + memo[cnt]
        
        # Determine the set S of allowed s values
        target = k - 1
        S = set()
        for s_val in range(1, max_s + 1):
            if memo[s_val] <= target:
                S.add(s_val)
        
        # Handle the case where n is 1
        if len(s) == 1 and s == "1":
            return 0
        
        digits = [int(c) for c in s]
        L = len(digits)
        # Initialize DP: [tight][count]
        dp = [[0] * (L + 2) for _ in range(2)]
        dp[1][0] = 1  # initial state: tight=1, count=0
        
        for i in range(L):
            curr_bit = digits[i]
            next_dp = [[0] * (L + 2) for _ in range(2)]
            for tight in [0, 1]:
                for count in range(L + 1):
                    if dp[tight][count] == 0:
                        continue
                    upper = curr_bit if tight else 1
                    for bit in range(0, upper + 1):
                        new_tight = tight and (bit == curr_bit)
                        new_count = count + (1 if bit else 0)
                        if new_count > L:
                            continue
                        next_dp[new_tight][new_count] = (next_dp[new_tight][new_count] + dp[tight][count]) % MOD
            dp = next_dp
        
        # Sum all counts where tight is 0 and count is in S
        total = 0
        for count in range(L + 1):
            if count in S:
                total = (total + dp[0][count]) % MOD
        
        return total % MOD