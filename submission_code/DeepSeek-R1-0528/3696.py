class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [dict() for _ in range(10)]
        res = 0
        for i in range(n):
            digit = int(s[i])
            new_dp = [dict() for _ in range(10)]
            if digit != 0:
                res += 1
            for d in range(1, 10):
                for rem, cnt in dp[d].items():
                    new_rem = (rem * 10 + digit) % d
                    new_dp[d][new_rem] = new_dp[d].get(new_rem, 0) + cnt
            if digit != 0:
                for rem, cnt in dp[digit].items():
                    if (10 * rem) % digit == 0:
                        res += cnt
                for d in range(1, 10):
                    r = digit % d
                    new_dp[d][r] = new_dp[d].get(r, 0) + 1
            else:
                for d in range(1, 10):
                    new_dp[d][0] = new_dp[d].get(0, 0) + 1
            dp = new_dp
        return res