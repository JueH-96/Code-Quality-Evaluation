class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        dp = {}

        def solve(z, o):
            if (z, o) in dp:
                return dp[(z, o)]

            if z > zero or o > one:
                return 0

            if z == zero and o == one:
                return 1

            ans = 0
            for i in range(1, min(zero - z + 1, limit + 1)):
                ans = (ans + solve(z + i, o)) % MOD
            for i in range(1, min(one - o + 1, limit + 1)):
                ans = (ans + solve(z, o + i)) % MOD

            dp[(z, o)] = ans
            return ans

        return solve(0, 0)