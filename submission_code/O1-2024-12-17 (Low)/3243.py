class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        """
        We want to count how many integers x in [start..finish] satisfy:
           1) x ends with the digits s,
           2) each digit of x is <= limit,
           3) s consists of digits (all <= limit) and has no leading zeros (given).

        Let S = int(s), and let L = len(s), M = 10^L.

        Any valid x can be written as x = K * 10^L + S, for some K >= 0.

        Condition: start <= K * M + S <= finish  -->  
                   K in [ (start - S + (M-1)) // M , (finish - S) // M ] (integer range)

        We also require each digit of x <= limit. Since the suffix s is already valid
        (its digits are <= limit by problem statement), we only need to ensure that
        the prefix K (in base 10) also has all digits <= limit.

        So we define:
            Kmin = max(0, ceil((start-S)/M))
            Kmax = floor((finish-S)/M)
        If Kmax < 0 or Kmin > Kmax then no valid x exist.

        Let f(n) be the count of integers in [0..n] whose digits are all <= limit.
        Then the answer for K in [Kmin..Kmax] is f(Kmax) - f(Kmin - 1).

        We'll write a digit-DP to compute f(n).
        """

        # Digit DP to count how many numbers from 0..n (inclusive) have each digit <= limit.
        def count_valid_up_to(n: int, limit: int) -> int:
            """
            Returns how many integers x in [0..n] have all digits of x <= limit.
            If n < 0, returns 0.
            """
            if n < 0:
                return 0
            # Convert n to a digit list
            digits = list(map(int, str(n)))

            # memo[pos][is_strict_smaller] = number of ways
            # pos: current index in digits (0-based)
            # is_strict_smaller: boolean (0 or 1) indicates whether we've already placed
            #                    a digit < digits[pos] in an earlier position,
            #                    which means we can freely choose up to limit in the current and subsequent positions.
            from functools import lru_cache

            @lru_cache(None)
            def dp(pos, is_strict_smaller, leading_zero):
                # If we've assigned all positions, we have formed one valid number
                if pos == len(digits):
                    return 1

                limit_digit = digits[pos]
                ways = 0
                # If we are already strictly smaller, we can use any digit up to limit.
                # Otherwise, we can only use up to limit_digit.
                max_digit = limit if is_strict_smaller else limit_digit

                for dig in range(0, max_digit + 1):
                    # We skip digits above limit
                    if dig > limit:
                        break

                    # If we are not strictly smaller yet and dig < limit_digit,
                    # we become strictly smaller for the next positions.
                    next_strict = is_strict_smaller or (dig < limit_digit)

                    # leading_zero means we haven't placed any nonzero digit so far
                    # it's valid to place 0 if leading_zero is True
                    # Actually, there's no restriction on leading zeros for counting

                    ways += dp(pos + 1, next_strict, leading_zero and (dig == 0))

                return ways

            return dp(0, False, True)

        # Main logic
        S_int = int(s)
        L = len(s)
        M = 10 ** L

        # If finish < S_int, then no number in the range can have suffix s
        if finish < S_int:
            return 0

        # Compute Kmin, Kmax
        # Kmin = ceil((start - S) / M), but not less than 0
        # Kmax = floor((finish - S) / M)
        # We'll do it carefully with integer math
        from math import ceil, floor

        def div_ceil(a, b):
            # returns ceil(a/b) for positive or negative a, b>0
            # careful with Python's division
            return (a + b - 1) // b if a > 0 else a // b

        r1 = start - S_int
        r2 = finish - S_int
        if r2 < 0:
            return 0

        # Kmax
        Kmax = r2 // M
        if Kmax < 0:
            return 0

        # Kmin
        # We want: K >= r1/M, so K >= ceil(r1/M)
        if r1 <= 0:
            Kmin = 0
        else:
            # ceil(r1/M) with integer arithmetic
            Kmin = (r1 + M - 1) // M

        if Kmin > Kmax:
            return 0

        # Count how many K in [Kmin..Kmax] have digits <= limit
        answer = count_valid_up_to(Kmax, limit) - count_valid_up_to(Kmin - 1, limit)
        return answer