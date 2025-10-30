class Solution:
    def __init__(self):
        MOD = 10**9 + 7
        max_n = 400
        self.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i-1] * i % MOD
        self.inv_fact = [1] * (max_n + 1)
        self.inv_fact[max_n] = pow(self.fact[max_n], MOD-2, MOD)
        for i in range(max_n-1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i+1] * (i+1) % MOD

    def comb(self, n, k):
        if n < 0 or k < 0 or k > n:
            return 0
        return self.fact[n] * self.inv_fact[k] % MOD * self.inv_fact[n - k] % MOD

    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        n = zero + one
        if n <= limit:
            return self.comb(n, zero) % MOD
        
        max_last = 2
        max_run_len = limit
        max_z = zero
        max_o = one
        dp = [[[[0] * (max_o + 1) for _ in range(max_z + 1)] for __ in range(max_run_len + 1)] for ___ in range(max_last)]
        
        if zero >= 1:
            dp[0][1][1][0] = 1
        if one >= 1:
            dp[1][1][0][1] = 1
        
        for s in range(2, n + 1):
            for z in range(0, zero + 1):
                o = s - z
                if o < 0 or o > one:
                    continue
                for last in [0, 1]:
                    for run_len in range(1, limit + 1):
                        current = dp[last][run_len][z][o]
                        if current == 0:
                            continue
                        
                        if last == 0:
                            if run_len < limit:
                                if z + 1 <= zero:
                                    new_z, new_o = z + 1, o
                                    new_run_len = run_len + 1
                                    dp[0][new_run_len][new_z][new_o] = (dp[0][new_run_len][new_z][new_o] + current) % MOD
                                if o + 1 <= one:
                                    new_z, new_o = z, o + 1
                                    new_run_len = 1
                                    dp[1][new_run_len][new_z][new_o] = (dp[1][new_run_len][new_z][new_o] + current) % MOD
                            else:
                                if o + 1 <= one:
                                    new_z, new_o = z, o + 1
                                    new_run_len = 1
                                    dp[1][new_run_len][new_z][new_o] = (dp[1][new_run_len][new_z][new_o] + current) % MOD
                        else:
                            if run_len < limit:
                                if o + 1 <= one:
                                    new_z, new_o = z, o + 1
                                    new_run_len = run_len + 1
                                    dp[1][new_run_len][new_z][new_o] = (dp[1][new_run_len][new_z][new_o] + current) % MOD
                                if z + 1 <= zero:
                                    new_z, new_o = z + 1, o
                                    new_run_len = 1
                                    dp[0][new_run_len][new_z][new_o] = (dp[0][new_run_len][new_z][new_o] + current) % MOD
                            else:
                                if z + 1 <= zero:
                                    new_z, new_o = z + 1, o
                                    new_run_len = 1
                                    dp[0][new_run_len][new_z][new_o] = (dp[0][new_run_len][new_z][new_o] + current) % MOD
        
        total = 0
        for last in [0, 1]:
            for run_len in range(1, limit + 1):
                total = (total + dp[last][run_len][zero][one]) % MOD
        return total