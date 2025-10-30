from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        from collections import Counter
        
        # Count zeroes separately (they only multiply the count).
        cnt = Counter(nums)
        zero_count = cnt.get(0, 0)
        if zero_count:
            del cnt[0]
        
        # We only care about sums up to r
        R = r
        # dp[s] = number of ways to pick from processed values to get sum = s
        dp = [1] + [0] * R
        
        # Process each distinct positive value v with multiplicity c
        for v, c in cnt.items():
            # If the value alone exceeds R, it can't contribute to any new sum
            if v > R:
                continue
            
            # Effective bound on how many of v we can take without exceeding R
            c_eff = min(c, R // v)
            
            prev = dp
            new_dp = [0] * (R + 1)
            window_size_shift = (c_eff + 1) * v
            
            # We split sums by their remainder mod v to do a sliding-window sum
            for rem in range(v):
                if rem > R:
                    break
                wsum = 0
                idx = rem
                # how many terms of the form rem + t*v fit in [0..R]
                T = (R - rem) // v
                
                # Slide a window of width (c_eff+1) over the sequence prev[rem], prev[rem+v], ...
                for t in range(T + 1):
                    # add the new term
                    wsum += prev[idx]
                    if wsum >= MOD:
                        wsum -= MOD
                    # remove the term that falls out of the window
                    if t > c_eff:
                        wsum -= prev[idx - window_size_shift]
                        if wsum < 0:
                            wsum += MOD
                    # record the number of ways to form sum = idx
                    new_dp[idx] = wsum
                    idx += v
            
            dp = new_dp
        
        # Sum up the ways for sums in [l..r]
        ans = sum(dp[l:]) % MOD
        # Each of those can be combined with any choice of zeros (0..zero_count)
        ans = ans * (zero_count + 1) % MOD
        return ans