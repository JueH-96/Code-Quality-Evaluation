class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        MOD = 10**9 + 7

        if s == t:
            same = 1
        else:
            same = 0
        
        diff_count = 0
        for i in range(1, n):
            temp = s[-i:] + s[:-i]
            if temp == t:
                diff_count += 1

        if k == 0:
            return int(s == t)

        if diff_count == 0:
            return int(s == t) if k % 2 == 0 else 0

        if same == 1:
            dp = [[0, 0] for _ in range(65)]
            dp[0][0] = 1
            for i in range(60):
                if (k >> i) & 1:
                    new_dp = [0, 0]
                    new_dp[0] = (dp[i][0] * same + dp[i][1] * (same - 1)) % MOD
                    new_dp[1] = (dp[i][0] * diff_count + dp[i][1] * (diff_count - 1)) % MOD
                    dp[i+1][0] = new_dp[0]
                    dp[i+1][1] = new_dp[1]
                else:
                    dp[i+1][0] = (dp[i][0] * same + dp[i][1] * (same - 1)) % MOD
                    dp[i+1][1] = (dp[i][0] * diff_count + dp[i][1] * (diff_count - 1)) % MOD
            return dp[60][0]
        else:
            dp = [[0, 0] for _ in range(65)]
            dp[0][1] = 1
            for i in range(60):
                if (k >> i) & 1:
                    new_dp = [0, 0]
                    new_dp[0] = (dp[i][0] * same + dp[i][1] * (same - 1)) % MOD
                    new_dp[1] = (dp[i][0] * diff_count + dp[i][1] * (diff_count - 1)) % MOD
                    dp[i+1][0] = new_dp[0]
                    dp[i+1][1] = new_dp[1]
                else:
                    dp[i+1][0] = (dp[i][0] * same + dp[i][1] * (same - 1)) % MOD
                    dp[i+1][1] = (dp[i][0] * diff_count + dp[i][1] * (diff_count - 1)) % MOD
            return dp[60][0]