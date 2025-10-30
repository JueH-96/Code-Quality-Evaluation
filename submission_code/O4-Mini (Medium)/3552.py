class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        # We build the largest n-digit palindrome divisible by k.
        # Let half = ceil(n/2). Define weights w[i] for prefix positions.
        # We do a digit-DP over prefix positions with mod state mod k,
        # to check feasibility, then greedily build largest digits.
        
        # Compute half-length of prefix
        half = (n + 1) // 2
        
        # Precompute powers of 10 mod k up to n
        pow10 = [0] * (n + 1)
        pow10[0] = 1 % k
        for i in range(1, n + 1):
            pow10[i] = (pow10[i - 1] * 10) % k
        
        # Compute weight w[i] for i in [1..half], 1-indexed
        w = [0] * (half + 1)
        for i in range(1, half + 1):
            # mirrored positions are i-1 and n-i
            if (n % 2 == 1) and (i == half):
                # middle digit only counts once
                w[i] = pow10[i - 1] % k
            else:
                w[i] = (pow10[n - i] + pow10[i - 1]) % k
        
        # Helper: rotate-right a k-bit mask x by t positions
        def rotate_right(x: int, t: int) -> int:
            # bits are 0..k-1
            if t == 0:
                return x
            # lower t bits of x
            low = x & ((1 << t) - 1)
            return ((x >> t) | (low << (k - t))) & ((1 << k) - 1)
        
        # DP masks: dp[pos] is a bitmask of length k,
        # bit m of dp[pos] = 1 means: if current mod at pos is m,
        # we can fill positions pos..half to reach total mod 0.
        dp = [0] * (half + 2)
        # Base: after half, mod must be 0 to succeed
        dp[half + 1] = 1 << 0
        
        # Build DP from pos = half down to 1
        for pos in range(half, 0, -1):
            mask = 0
            wi = w[pos]
            # possible digits at pos
            if pos == 1:
                d_start, d_end = 1, 9
            else:
                d_start, d_end = 0, 9
            nxt = dp[pos + 1]
            # For each digit, rotate the next-mask and OR
            for d in range(d_start, d_end + 1):
                t = (d * wi) % k
                mask |= rotate_right(nxt, t)
            dp[pos] = mask
        
        # If no solution from mod 0 at pos 1, return empty
        if ((dp[1] >> 0) & 1) == 0:
            return ""
        
        # Greedily build the prefix
        prefix = [0] * (half + 1)  # 1-indexed
        cur_mod = 0
        for pos in range(1, half + 1):
            wi = w[pos]
            if pos == 1:
                digits = range(9, 0, -1)
            else:
                digits = range(9, -1, -1)
            found = False
            for d in digits:
                new_mod = (cur_mod + d * wi) % k
                # check feasibility from pos+1 with new_mod
                if ((dp[pos + 1] >> new_mod) & 1) == 1:
                    prefix[pos] = d
                    cur_mod = new_mod
                    found = True
                    break
            # must find one
            if not found:
                # should not happen
                return ""
        
        # Build result string
        # prefix digits
        left = ''.join(str(prefix[i]) for i in range(1, half + 1))
        # mirrored suffix
        if n % 2 == 0:
            right = left[::-1]
        else:
            right = left[:-1][::-1]
        return left + right