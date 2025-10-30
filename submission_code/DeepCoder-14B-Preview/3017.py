class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def helper(x, n, k):
            s = list(map(int, str(x)))
            if len(s) < n:
                return 0
            if len(s) > n:
                return helper(10**n - 1, n, k)
            dp = [{} for _ in range(n + 1)]
            dp[0] = {(0, 0, 0, True): 1}
            for pos in range(n):
                current_dp = dp[pos]
                for state in current_dp:
                    even, odd, mod, tight = state
                    count = current_dp[state]
                    max_d = s[pos] if tight else 9
                    if pos == 0:
                        start = 1
                    else:
                        start = 0
                    for d in range(start, max_d + 1):
                        new_tight = tight and (d == max_d)
                        new_even = even + (1 if d % 2 == 0 else 0)
                        new_odd = odd + (1 if d % 2 != 0 else 0)
                        new_mod = (mod * 10 + d) % k
                        key = (new_even, new_odd, new_mod, new_tight)
                        if key not in dp[pos + 1]:
                            dp[pos + 1][key] = 0
                        dp[pos + 1][key] += count
            total = 0
            for state in dp[n]:
                even, odd, mod, _ = state
                if even == odd and mod == 0:
                    total += dp[n][state]
            return total
        
        total = 0
        max_digits = len(str(high))
        for n in range(2, max_digits + 1, 2):
            lower = 10 ** (n - 1)
            upper = (10 ** n) - 1
            a = max(low, lower)
            b = min(high, upper)
            if a > b:
                continue
            count_b = helper(b, n, k)
            count_a_minus_1 = helper(a - 1, n, k) if a > 1 else 0
            total += (count_b - count_a_minus_1)
        return total