class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        N = len(num)
        freq = [0] * 10
        S = 0
        for c in num:
            d = int(c)
            freq[d] += 1
            S += d
        
        if S % 2 != 0:
            return 0
        T = S // 2
        C = (N + 1) // 2  # Number of even positions
        
        # DP[i][j][k] = number of ways to choose j digits with sum k using digits 0 to i
        DP = [[[0 for _ in range(T + 1)] for _ in range(C + 1)] for _ in range(11)]
        DP[0][0][0] = 1
        
        for d in range(10):
            for j in range(C + 1):
                for k in range(T + 1):
                    for f in range(min(freq[d] + 1, j + 1)):
                        if f * d <= k:
                            DP[d + 1][j][k] += DP[d][j - f][k - f * d]
                            if DP[d + 1][j][k] >= MOD:
                                DP[d + 1][j][k] -= MOD
                        else:
                            break
            # Optimization: reset the inner loops to zero for the next digit
            for j in range(C + 1):
                for k in range(T + 1):
                    DP[d + 1][j][k] %= MOD
        
        ways = DP[10][C][T]
        
        # Calculate factorial of counts
        fact = [1]
        for i in range(1, N + 1):
            fact.append(fact[-1] * i % MOD)
        
        # Calculate inverse factorial
        inv_fact = [pow(fact[i], MOD - 2, MOD) for i in range(N + 1)]
        
        # Calculate the total number of distinct permutations
        total_perm = fact[N]
        for f in freq:
            total_perm = total_perm * inv_fact[f] % MOD
        
        # Calculate arrangements for even and odd positions
        even_perm = fact[C]
        odd_perm = fact[N - C]
        for d in range(10):
            for f in range(min(freq[d] + 1, C + 1)):
                if C - f < 0:
                    continue
                even_freq = [0] * 10
                odd_freq = [0] * 10
                even_freq[d] = f
                odd_freq[d] = freq[d] - f
                # Calculate arrangements for even positions
                arr_even = even_perm * inv_fact[f] % MOD
                # Calculate arrangements for odd positions
                arr_odd = odd_perm * inv_fact[freq[d] - f] % MOD
                # Accumulate the result
                ways = (ways + arr_even * arr_odd) % MOD
        
        return ways