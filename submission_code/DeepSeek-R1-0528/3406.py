class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        
        dp0 = [[[0] * limit for _ in range(one+1)] for __ in range(zero+1)]
        dp1 = [[[0] * limit for _ in range(one+1)] for __ in range(zero+1)]
        
        if zero >= 1:
            dp0[1][0][0] = 1
        if one >= 1:
            dp1[0][1][0] = 1
        
        for z in range(zero+1):
            for o in range(one+1):
                if z >= 1:
                    total_prev = sum(dp1[z-1][o]) % mod
                    dp0[z][o][0] = (dp0[z][o][0] + total_prev) % mod
                    for kcon in range(2, limit+1):
                        prev_index = kcon - 2
                        dp0[z][o][kcon-1] = (dp0[z][o][kcon-1] + dp0[z-1][o][prev_index]) % mod
                if o >= 1:
                    total_prev = sum(dp0[z][o-1]) % mod
                    dp1[z][o][0] = (dp1[z][o][0] + total_prev) % mod
                    for kcon in range(2, limit+1):
                        prev_index = kcon - 2
                        dp1[z][o][kcon-1] = (dp1[z][o][kcon-1] + dp1[z][o-1][prev_index]) % mod
        
        result = (sum(dp0[zero][one]) + sum(dp1[zero][one])) % mod
        return result