class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        import functools

        # Let factor = 10^(len(s)) so that any x ending with s can be written as:
        #   x = k * factor + tail   where tail = int(s)
        factor = 10 ** len(s)
        tail = int(s)
        
        # If finish is less than the smallest possible x (which is tail itself),
        # then no powerful integer in the range exists.
        if finish < tail:
            return 0

        # We want x = k * factor + tail to lie in [start, finish]. This gives:
        #   start <= k * factor + tail <= finish
        # Hence, for k we have:
        #   k >= ceil((start - tail) / factor)  and  k <= (finish - tail) // factor.
        # Also note: k is a nonnegative integer.
        if start <= tail:
            kLow = 0
        else:
            # Compute ceiling of (start - tail)/factor using integer arithmetic.
            kLow = (start - tail + factor - 1) // factor
        
        kHigh = (finish - tail) // factor
        
        if kHigh < kLow:
            return 0

        # We now need to count how many integers k with kLow <= k <= kHigh
        # have the property that every digit in k is at most limit.
        # (The tail part "s" is already given to have digits <= limit.)
        # Because k can be large (and its range may span many values), we use digit DP.

        def count_valid(n: int, limit: int) -> int:
            # Counts numbers in [0, n] whose decimal representation (without
            # extraneous leading zeros) contains only digits <= limit.
            if n < 0:
                return 0
            digits = list(map(int, str(n)))
            
            @functools.lru_cache(maxsize=None)
            def dp(pos: int, tight: bool, started: bool) -> int:
                # pos: current position in the digit array (from 0 to len(digits))
                # tight: whether we are still bound by the prefix of n
                # started: whether we have started placing a nonzero digit.
                if pos == len(digits):
                    return 1  # One valid number is formed (0 counts when not started)
                # If we are "tight", the next digit can be at most digits[pos].
                # But in any case, our digit must not exceed the limit.
                up = digits[pos] if tight else limit
                up = min(up, limit)
                total = 0
                for d in range(0, up + 1):
                    new_tight = tight and (d == digits[pos])
                    new_started = started or (d != 0)
                    total += dp(pos + 1, new_tight, new_started)
                return total
            
            return dp(0, True, False)
        
        # Count valid k values in the interval [kLow, kHigh].
        # count_valid(n, limit) counts valid numbers in [0, n], so the answer is:
        #   count_valid(kHigh, limit) - count_valid(kLow - 1, limit)
        return count_valid(kHigh, limit) - count_valid(kLow - 1, limit)