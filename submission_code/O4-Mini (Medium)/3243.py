from functools import lru_cache

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Convert suffix to integer and get its length
        s_int = int(s)
        m = len(s)
        # Precompute 10^m
        pow10m = 10 ** m
        # The maximum total length of finish
        max_len = len(str(finish))
        # Result counter
        res = 0
        
        # Handle the case with no prefix (L = 0): the number is exactly s
        if start <= s_int <= finish:
            res += 1
        
        # For prefixes of length L = 1 .. (max_len - m)
        max_L = max_len - m
        for L in range(1, max_L + 1):
            # Compute the range of prefix p so that p*10^m + s_int in [start, finish]
            # p >= ceil((start - s_int) / 10^m)
            # p <= floor((finish - s_int) / 10^m)
            lo_num = start - s_int
            hi_num = finish - s_int
            low_p = (lo_num + pow10m - 1) // pow10m    # ceil division
            high_p = hi_num // pow10m                # floor division
            
            # Enforce that p has exactly L digits: [10^(L-1) .. 10^L - 1]
            min_p = 10 ** (L - 1)
            max_p = 10 ** L - 1
            p_lo = max(min_p, low_p)
            p_hi = min(max_p, high_p)
            if p_hi < p_lo:
                continue
            
            # Digit-DP to count how many L-digit numbers in [p_lo..p_hi] have all digits <= limit
            def count_upto(n: int) -> int:
                # If n is below the smallest L-digit number, no valid p
                if n < min_p:
                    return 0
                # Make n_str of length L by zero-padding on the left
                ns = str(n).rjust(L, '0')
                
                @lru_cache(None)
                def dp(pos: int, tight: bool) -> int:
                    """
                    Count ways to fill digits pos..L-1 given tight constraint
                    pos: current digit position (0-based from left)
                    tight: whether previous digits matched the prefix of n
                    """
                    if pos == L:
                        # filled all positions
                        return 1
                    total = 0
                    # Bound for this digit
                    bound = int(ns[pos]) if tight else limit
                    if bound > limit:
                        bound = limit
                    # Determine lower limit for this position
                    if pos == 0:
                        d_start = 1  # no leading zero
                    else:
                        d_start = 0
                    # Try all digits in [d_start..bound]
                    for d in range(d_start, bound + 1):
                        new_tight = tight and (d == int(ns[pos]))
                        total += dp(pos + 1, new_tight)
                    return total
                
                return dp(0, True)
            
            # Count valid prefixes in [p_lo..p_hi]
            cnt_hi = count_upto(p_hi)
            cnt_lo = count_upto(p_lo - 1)
            res += (cnt_hi - cnt_lo)
        
        return res