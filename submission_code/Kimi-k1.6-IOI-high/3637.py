class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        cnt = [0] * 10
        for c in num:
            cnt[int(c)] += 1
        
        S = sum(int(c) for c in num)
        n = len(num)
        m = (n + 1) // 2
        k = n // 2
        
        if S % 2 != 0:
            return 0
        
        T = S // 2
        
        dp = [[0] * (T + 1) for _ in range(m + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            if cnt[d] == 0:
                continue
            c = cnt[d]
            new_dp = [[0] * (T + 1) for _ in range(m + 1)]
            for sum_x in range(m + 1):
                for sum_val in range(T + 1):
                    if dp[sum_x][sum_val] == 0:
                        continue
                    for x in range(0, c + 1):
                        new_sum_x = sum_x + x
                        new_sum_val = sum_val + d * x
                        if new_sum_x > m or new_sum_val > T:
                            continue
                        term = dp[sum_x][sum_val] * inv_fact[x] % MOD
                        term = term * inv_fact[c - x] % MOD
                        new_dp[new_sum_x][new_sum_val] = (new_dp[new_sum_x][new_sum_val] + term) % MOD
            dp = new_dp
        
        ans = dp[m][T] * fact[m] % MOD
        ans = ans * fact[k] % MOD
        return ans