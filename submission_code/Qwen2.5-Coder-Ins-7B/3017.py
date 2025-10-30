class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful_numbers(n):
            if n < 10:
                return 0
            digits = []
            while n > 0:
                digits.append(n % 10)
                n //= 10
            digits.reverse()
            n = len(digits)
            if n % 2 != 0:
                return 0
            half = n // 2
            dp = [[[[[0] * 20 for _ in range(2)] for _ in range(half + 1)] for _ in range(half + 1)] for _ in range(2)]
            dp[0][0][0][0][0] = 1
            for i in range(n):
                new_dp = [[[[[0] * 20 for _ in range(2)] for _ in range(half + 1)] for _ in range(half + 1)] for _ in range(2)]
                for j in range(half + 1):
                    for l in range(half + 1):
                        for m in range(2):
                            for p in range(20):
                                for q in range(2):
                                    if dp[j][l][m][p][q] == 0:
                                        continue
                                    if i < n - 1:
                                        new_dp[j][l][m][p][q] += dp[j][l][m][p][q] * (digits[i] + 1)
                                        new_dp[j][l][m][p][q] %= k
                                    else:
                                        if j == l and (p + digits[i]) % 2 == 0 and q == (digits[i] % 2):
                                            new_dp[j][l][m][p][q] += dp[j][l][m][p][q]
                                            new_dp[j][l][m][p][q] %= k
            return sum(dp[j][l][m][p][q] for j in range(half + 1) for l in range(half + 1) for m in range(2) for p in range(20) for q in range(2)) % k
        
        return (count_beautiful_numbers(high + 1) - count_beautiful_numbers(low)) % k