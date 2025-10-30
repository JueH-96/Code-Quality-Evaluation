mod = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        total_sum = sum(int(d) for d in num)
        if total_sum % 2 != 0:
            return 0
        
        n = len(num)
        E = (n + 1) // 2
        O = n - E
        half = total_sum // 2
        
        freq = [0] * 10
        for d in num:
            freq[int(d)] += 1
        
        max_val = 80
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = fact[i - 1] * i % mod
        inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
        for i in range(max_val, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % mod
        
        base = fact[E] * fact[O] % mod
        
        dp = [[0] * (half + 1) for _ in range(E + 1)]
        dp[0][0] = 1
        
        for d in range(10):
            cnt = freq[d]
            if cnt == 0:
                continue
            ndp = [[0] * (half + 1) for _ in range(E + 1)]
            for i in range(E + 1):
                for j in range(half + 1):
                    if dp[i][j] == 0:
                        continue
                    max_k = min(cnt, E - i)
                    for k in range(max_k + 1):
                        new_j = j + k * d
                        if new_j > half:
                            break
                        factor = inv_fact[k] * inv_fact[cnt - k] % mod
                        ndp[i + k][new_j] = (ndp[i + k][new_j] + dp[i][j] * factor) % mod
            dp = ndp
        
        res = base * dp[E][half] % mod
        return res