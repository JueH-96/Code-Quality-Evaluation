class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp0 = [[0] * (one + 1) for _ in range(zero + 1)]
        dp1 = [[0] * (one + 1) for _ in range(zero + 1)]
        
        # Initialize runs starting with zeros or ones
        for k in range(1, min(limit, zero) + 1):
            dp0[k][0] = 1
        for k in range(1, min(limit, one) + 1):
            dp1[0][k] = 1
        
        for z in range(zero + 1):
            for o in range(one + 1):
                if z == 0 and o == 0:
                    continue
                # Compute dp0[z][o]
                total0 = 0
                max_kz = min(limit, z)
                for k in range(1, max_kz + 1):
                    if z >= k:
                        total0 += dp1[z - k][o]
                        if total0 >= MOD:
                            total0 -= MOD
                dp0[z][o] = (dp0[z][o] + total0) % MOD
                
                # Compute dp1[z][o]
                total1 = 0
                max_ko = min(limit, o)
                for k in range(1, max_ko + 1):
                    if o >= k:
                        total1 += dp0[z][o - k]
                        if total1 >= MOD:
                            total1 -= MOD
                dp1[z][o] = (dp1[z][o] + total1) % MOD
        
        return (dp0[zero][one] + dp1[zero][one]) % MOD