class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # If target is 1, we can only use numbers >= 1
        # and no two numbers can sum to 1, so we just use 1, 2, 3, ..., n
        if target == 1:
            return (n * (n + 1) // 2) % MOD
        
        # We can use at most target//2 numbers from the range [1, target-1]
        # because for each x < target/2, we can't use both x and (target-x)
        half = target // 2
        
        if n <= half:
            # We can use numbers 1, 2, ..., n
            return (n * (n + 1) // 2) % MOD
        else:
            # Use numbers 1, 2, ..., half and then target, target+1, ..., target+(n-half-1)
            # Sum of 1 to half: half * (half + 1) // 2
            # Sum of target to target+(n-half-1): 
            # = target + (target+1) + ... + (target+n-half-1)
            # = n-half terms starting from target
            # = (n-half) * target + (0 + 1 + ... + (n-half-1))
            # = (n-half) * target + (n-half-1) * (n-half) // 2
            
            sum1 = half * (half + 1) // 2
            remaining = n - half
            sum2 = remaining * target + (remaining - 1) * remaining // 2
            
            return (sum1 + sum2) % MOD