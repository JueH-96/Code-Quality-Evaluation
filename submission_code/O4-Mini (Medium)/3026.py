class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Number of picks we can take from [1, 2, ..., target-1] without any pair summing to target
        # We greedily take the smaller of each forbidden pair (x, target-x).
        m = min(n, target // 2)
        # If we only need <= m numbers, they are just 1,2,...,n
        if n <= m:
            return (n * (n + 1) // 2) % MOD
        
        # Sum of the first m natural numbers
        sum_small = m * (m + 1) // 2
        # We still need k more numbers, which must start from `target` upwards
        k = n - m
        # Sum of k numbers starting at `target`:
        # target + (target+1) + ... + (target + k - 1)
        # = k*target + (0 + 1 + ... + (k-1)) = k*target + k*(k-1)/2
        sum_large = k * target + (k * (k - 1) // 2)
        
        total = (sum_small + sum_large) % MOD
        return total