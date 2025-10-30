class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # If target is odd, we can use all numbers from 1 to n
        if target % 2 == 1:
            return (n * (n + 1) // 2) % MOD
        
        # If target is even, we can't use target // 2
        # So we use numbers from 1 to n+1, skipping target // 2
        else:
            return ((n + 1) * (n + 2) // 2 - target // 2) % MOD