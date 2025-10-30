class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        nlen = len(s)
        max_val = 1000
        step = [0] * (max_val + 1)
        for i in range(2, max_val + 1):
            p = bin(i).count('1')
            step[i] = 1 + step[p]
        
        valid_p = set()
        for p in range(1, nlen + 1):
            if p <= max_val and step[p] <= k - 1:
                valid_p.add(p)
        
        dp = [[0] * (nlen + 1) for _ in range(2)]
        dp[1][0] = 1
        
        for i in range(nlen):
            new_dp = [[0] * (nlen + 1) for _ in range(2)]
            for tight in (0, 1):
                for cnt in range(nlen + 1):
                    ways = dp[tight][cnt]
                    if ways == 0:
                        continue
                    up = int(s[i]) if tight else 1
                    for d in (0, 1):
                        if d > up:
                            continue
                        new_tight = tight and (d == up)
                        new_cnt = cnt + (1 if d == 1 else 0)
                        if new_cnt <= nlen:
                            new_dp[new_tight][new_cnt] = (new_dp[new_tight][new_cnt] + ways) % mod
            dp = new_dp
        
        res = 0
        for tight in (0, 1):
            for cnt in range(nlen + 1):
                if tight == 0 and cnt in valid_p:
                    res = (res + dp[tight][cnt]) % mod
        return res