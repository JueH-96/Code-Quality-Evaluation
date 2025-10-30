class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def count_up_to(num: str) -> int:
            n = len(num)
            dp = [[0] * 10 for _ in range(n + 1)]
            dp[0][0] = 1
            
            for i in range(1, n + 1):
                for d in range(10):
                    for next_d in [d - 1, d + 1]:
                        if 0 <= next_d <= 9:
                            dp[i][d] += dp[i - 1][next_d]
                            dp[i][d] %= MOD
            
            count = 0
            for d in range(1, 10):
                count += dp[n][d]
                count %= MOD
            
            for i in range(1, n):
                for d in range(1, 10):
                    count += dp[i][d]
                    count %= MOD
            
            for i in range(1, 10):
                if i <= int(num[0]):
                    count += dp[n - 1][i]
                    count %= MOD
            
            for i in range(1, n):
                d = int(num[i - 1])
                for next_d in [d - 1, d + 1]:
                    if 0 <= next_d <= 9 and next_d < int(num[i]):
                        count += dp[n - i][next_d]
                        count %= MOD
                if int(num[i - 1]) - 1 == int(num[i]) or int(num[i - 1]) + 1 == int(num[i]):
                    count += 1
                    count %= MOD
                else:
                    break
            
            return count
        
        return (count_up_to(high) - count_up_to(str(int(low) - 1)) + MOD) % MOD