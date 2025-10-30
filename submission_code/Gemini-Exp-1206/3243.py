class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count(n: int, limit: int, s: str) -> int:
            t = str(n)
            m = len(t)
            k = len(s)
            dp = [[0] * (m + 1) for _ in range(m + 1)]
            dp[0][0] = 1
            for i in range(1, m - k + 1):
                for j in range(i):
                    for d in range(limit + 1):
                        if i == 1 and d == 0:
                            continue
                        if j == 0:
                            if d < int(t[i - 1]):
                                dp[i][1] += dp[i - 1][j]
                            elif d == int(t[i - 1]):
                                dp[i][0] += dp[i - 1][j]
                        else:
                            dp[i][1] += dp[i - 1][j]
            ans = 0
            for i in range(m - k + 1):
                ok = True
                for j in range(k):
                    if int(s[j]) > limit or (i + j < m and int(s[j]) < int(t[i + j])):
                        ok = False
                        break
                    if i + j < m and int(s[j]) > int(t[i + j]):
                        break
                if ok:
                    ans += dp[i][0]
                    if i < m - k:
                        ans += dp[i][1]
            return ans

        return count(finish, limit, s) - count(start - 1, limit, s)