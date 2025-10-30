MOD = 10**9 + 7
M = 10000

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        min_ops = [0] * (M + 1)
        min_ops[1] = 0
        for i in range(2, M + 1):
            next_i = bin(i).count('1')
            min_ops[i] = 1 + min_ops[next_i]
        
        S = set()
        for i in range(1, M + 1):
            if min_ops[i] <= k - 1:
                S.add(i)
        
        n_len = len(s)
        dp = [[0] * (n_len + 1) for _ in range(2)]
        dp[1][0] = 1
        
        for char in s:
            current_digit = int(char)
            new_dp = [[0] * (n_len + 1) for _ in range(2)]
            for tight in range(2):
                for count in range(n_len + 1):
                    if dp[tight][count] == 0:
                        continue
                    high = current_digit if tight else 1
                    for d in range(0, high + 1):
                        new_tight = tight and (d == high)
                        new_count = count + (1 if d == 1 else 0)
                        if new_count <= n_len:
                            new_dp[new_tight][new_count] = (new_dp[new_tight][new_count] + dp[tight][count]) % MOD
            dp = new_dp
        
        total = 0
        for count in range(n_len + 1):
            if count in S:
                total = (total + dp[0][count]) % MOD
        return total