class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        def solve(target, start, current_sum, memo):
            if (target, start) in memo:
                return memo[(target, start)]
            
            if target == current_sum:
                return 1
            if target < current_sum:
                return 0
            
            count = 0
            for i in range(start, int(target**(1/x)) + 1):
                count = (count + solve(target, i + 1, current_sum + i**x, memo)) % MOD
            
            memo[(target, start)] = count
            return count
        
        memo = {}
        return solve(n, 1, 0, memo)