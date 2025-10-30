class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        import math
        
        m = len(target)
        # bitmask for all targets covered
        FULL = (1 << m) - 1
        
        # Precompute the LCM for every non-zero subset of targets
        lcms = [1] * (FULL + 1)
        for mask in range(1, FULL + 1):
            # isolate lowest set bit
            lb = mask & -mask
            idx = lb.bit_length() - 1
            prev = mask ^ lb
            if prev == 0:
                lcms[mask] = target[idx]
            else:
                prev_lcm = lcms[prev]
                t = target[idx]
                lcms[mask] = prev_lcm // math.gcd(prev_lcm, t) * t
        
        # dp[mask] = minimum cost to cover exactly the subset "mask" of targets
        INF = 10**30
        dp = [INF] * (FULL + 1)
        dp[0] = 0
        
        # For each number in nums, we can choose to use it (once) to cover
        # any submask of the uncovered targets, paying the increment cost.
        for n in nums:
            old = dp
            dp = old.copy()
            for covered in range(FULL + 1):
                base_cost = old[covered]
                if base_cost >= INF:
                    continue
                # available targets not yet covered
                avail = FULL ^ covered
                # enumerate all non-empty submasks of avail
                s = avail
                while s:
                    l = lcms[s]
                    rem = n % l
                    cost = 0 if rem == 0 else (l - rem)
                    new_mask = covered | s
                    if dp[new_mask] > base_cost + cost:
                        dp[new_mask] = base_cost + cost
                    s = (s - 1) & avail
        
        return dp[FULL]