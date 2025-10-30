class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7

        dp = {}

        def solve(performers, stages):
            if (performers, stages) in dp:
                return dp[(performers, stages)]

            if performers == 0:
                return 1 if stages == 0 else 0
            
            if stages == 0:
                return 0

            ans = 0
            for i in range(performers + 1):
                ans = (ans + (pow(y, stages, MOD) * solve(performers - i, stages - 1)) % MOD) % MOD
            
            dp[(performers, stages)] = ans
            return ans

        
        ways_to_assign_stages = pow(x, n, MOD)
        
        ways_to_score = 0
        for num_stages in range(1, min(n, x) + 1):
            ways_to_score = (ways_to_score + solve(n, num_stages)) % MOD
        
        return ways_to_score