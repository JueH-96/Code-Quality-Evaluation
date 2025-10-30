class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        # Number of "small" safe picks from [1..target-1] is k = floor((target-1)/2)
        k = (target - 1) // 2
        # If we only need n <= k, we just take 1..n
        if n <= k:
            return (n * (n + 1) // 2) % MOD
        
        # Otherwise we take all of 1..k, then r = n-k more from the "large" safe numbers
        sum_small = k * (k + 1) // 2
        r = n - k
        
        # If target is even, the next safe number after 1..k is target/2,
        # then the sequence target, target+1, ...
        if target % 2 == 0:
            m = target // 2
            # sum of first r safe picks beyond 1..k:
            #   = m + sum_{i=0..r-2} (target + i)
            #   = m*(2r-1) + (r-1)*(r-2)/2
            sum_large = m * (2 * r - 1) + (r - 1) * (r - 2) // 2
        else:
            # If target is odd, the next safe pick after 1..k is target itself,
            # then target+1, target+2, ...
            # sum_large = sum_{i=0..r-1} (target + i)
            #           = r*target + r*(r-1)/2
            sum_large = r * target + r * (r - 1) // 2
        
        return (sum_small + sum_large) % MOD