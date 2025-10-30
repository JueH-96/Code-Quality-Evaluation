class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # We can use numbers from 1 to target//2
        limit = target // 2
        
        if n <= limit:
            # We can use 1, 2, ..., n
            return (n * (n + 1) // 2) % MOD
        else:
            # We use 1, 2, ..., limit, then target, target+1, ..., target+(n-limit-1)
            sum1 = limit * (limit + 1) // 2
            remaining = n - limit
            sum2 = remaining * target + remaining * (remaining - 1) // 2
            return (sum1 + sum2) % MOD