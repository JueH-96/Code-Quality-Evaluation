class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = int(s, 2)
        if n <= 1:
            return 0
        
        memo = {1: 0}
        
        def steps(x):
            if x in memo:
                return memo[x]
            c = bin(x).count('1')
            res = 1 + steps(c)
            memo[x] = res
            return res
        
        count = 0
        for x in range(1, n):
            if steps(x) <= k:
                count += 1
        
        return count % MOD