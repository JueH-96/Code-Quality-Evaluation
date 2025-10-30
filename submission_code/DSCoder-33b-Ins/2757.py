class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7
        n = len(num2)
        dp = [[[0]*max_sum for _ in range(2)] for _ in range(n)]
        prefix = [[[0]*max_sum for _ in range(2)] for _ in range(n)]

        def add(x, y):
            x += y
            if x >= MOD:
                x -= MOD
            return x

        def sub(x, y):
            x -= y
            if x < 0:
                x += MOD
            return x

        def solve(idx, tight, sum, num):
            if idx == n:
                return sum >= min_sum and sum <= max_sum
            if prefix[idx][tight][sum] != -1:
                return prefix[idx][tight][sum]
            limit = 9 if tight else int(num[idx])
            ans = 0
            for digit in range(limit+1):
                curr_tight = tight and digit == int(num[idx])
                if sum + digit > max_sum:
                    break
                ans = add(ans, solve(idx+1, curr_tight, sum+digit, num))
            prefix[idx][tight][sum] = ans
            return ans

        num1 = num1.zfill(n)
        num2 = num2.zfill(n)
        ans = sub(solve(0, 1, 0, num2), solve(0, 1, 0, num1))
        for i in range(n):
            for j in range(2):
                for k in range(max_sum):
                    prefix[i][j][k] = -1
        return ans