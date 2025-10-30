MOD = 1_000_000_007


class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        # dp0[i][j] – ends with 0-block, used i zeros, j ones
        # dp1[i][j] – ends with 1-block
        dp0 = [[0] * (one + 1) for _ in range(zero + 1)]
        dp1 = [[0] * (one + 1) for _ in range(zero + 1)]

        # first block – 0
        for t in range(1, min(limit, zero) + 1):
            dp0[t][0] = 1
        # first block – 1
        for t in range(1, min(limit, one) + 1):
            dp1[0][t] = 1

        for i in range(zero + 1):
            for j in range(one + 1):
                v0 = dp0[i][j]
                if v0:
                    max_add_ones = min(limit, one - j)
                    for t in range(1, max_add_ones + 1):
                        dp1[i][j + t] = (dp1[i][j + t] + v0) % MOD

                v1 = dp1[i][j]
                if v1:
                    max_add_zeros = min(limit, zero - i)
                    for t in range(1, max_add_zeros + 1):
                        dp0[i + t][j] = (dp0[i + t][j] + v1) % MOD

        return (dp0[zero][one] + dp1[zero][one]) % MOD