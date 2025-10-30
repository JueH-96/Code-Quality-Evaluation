class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        
        # If target is 1, we can only use numbers >= target (which is 1)
        # So we use 1, 2, 3, ..., n
        if target == 1:
            return (n * (n + 1) // 2) % MOD
        
        # Numbers we can use without restriction: 1, 2, ..., target//2
        # and target, target+1, target+2, ...
        
        half_target = target // 2
        
        if n <= half_target:
            # We can use the first n positive integers
            return (n * (n + 1) // 2) % MOD
        else:
            # Use 1, 2, ..., half_target, then target, target+1, ...
            # Sum of 1 to half_target
            sum1 = (half_target * (half_target + 1) // 2) % MOD
            
            # Remaining numbers needed
            remaining = n - half_target
            
            # Sum of target, target+1, ..., target+remaining-1
            # This is sum from target to target+remaining-1
            # = remaining * target + (0 + 1 + ... + (remaining-1))
            # = remaining * target + remaining*(remaining-1)/2
            sum2 = (remaining * target + remaining * (remaining - 1) // 2) % MOD
            
            return (sum1 + sum2) % MOD