class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        def helper(target, power, start):
            if target == 0:
                return 1
            if target < 0:
                return 0
            ways = 0
            for i in range(start, int(target**(1/x)) + 1):
                ways += helper(target - i**x, power, i + 1)
            return ways % MOD
        
        return helper(n, x, 1)