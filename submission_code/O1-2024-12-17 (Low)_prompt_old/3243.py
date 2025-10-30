class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # We want to count how many integers x in [start..finish] satisfy:
        # 1) x ends with the string s
        # 2) every digit in x is <= limit
        # Equivalently, let suffixVal = int(s). If x = p * (10^len(s)) + suffixVal,
        # then we need start <= x <= finish and each digit of x <= limit.
        # Since s only contains digits <= limit, the suffix part is already valid.
        # We only need to ensure that the prefix p (which may be 0, meaning no extra digits)
        # also has digits <= limit, and that x is in [start..finish].
        #
        # So, from start <= p*10^len(s) + suffixVal <= finish, we get:
        # start - suffixVal <= p*10^len(s) <= finish - suffixVal
        # => (start - suffixVal)/10^len(s) <= p <= (finish - suffixVal)//10^len(s)  (integer bounds)
        #
        # Then we only need to count all p in [low..high] whose digits are all <= limit.
        # We'll do that using a standard "digit DP" approach to count how many numbers in [0..X]
        # have all digits <= limit. Call that function f(X, limit). Then the result for
        # p in [low..high] is f(high, limit) - f(low-1, limit) (if low > 0).
        #
        # Steps:
        # 1) If suffixVal > finish, answer is 0 (can't fit in [start..finish]).
        # 2) Let powerOfTen = 10^len(s).
        # 3) Define low = ceil((start - suffixVal)/powerOfTen), but also bounded below by 0, since p >= 0.
        # 4) Define high = floor((finish - suffixVal)/powerOfTen).
        # 5) If low > high, return 0.
        # 6) The answer is f(high, limit) - f(low-1, limit) where f is the digit-DP function
        #    (if low > 0, else it's f(high, limit)).
        #
        # We'll implement a helper digit-DP: count_up_to(X, limit) that returns how many integers
        # in the range [0..X] have all digits <= limit.

        # Convert s to integer:
        suffix_val = int(s)
        len_s = len(s)
        power_of_ten = 10 ** len_s

        # If suffix_val > finish, no valid numbers
        if suffix_val > finish:
            return 0

        # Compute the range for p
        # p * power_of_ten + suffix_val >= start  => p >= (start - suffix_val)/power_of_ten
        # p * power_of_ten + suffix_val <= finish => p <= (finish - suffix_val)/power_of_ten
        import math

        # lower bound
        # we do a ceiling for the fraction (start - suffix_val + (power_of_ten - 1)) // power_of_ten
        # but we can also do integer division carefully
        if start <= suffix_val:
            # then p can be 0 if suffix_val >= start
            low = 0
        else:
            # We want the smallest integer p such that p >= (start - suffix_val)/power_of_ten
            # (start - suffix_val + power_of_ten - 1) // power_of_ten for positive denominators,
            # but handle negative carefully if (start < suffix_val). We'll unify by max(0,).
            low = (start - suffix_val + power_of_ten - 1) // power_of_ten
            if low < 0:
                low = 0

        # upper bound
        # p <= (finish - suffix_val)//power_of_ten
        high = (finish - suffix_val) // power_of_ten
        if high < 0:
            return 0  # then no valid p if even the largest negative doesn't fit

        if low > high:
            return 0

        # Now we define a digit DP to count how many integers p in [0..N] have all digits <= limit.
        def count_valid_digits_up_to(N: int, limit: int) -> int:
            """Return how many integers in [0..N] have each digit <= limit."""
            if N < 0:
                return 0
            # Convert N to digit array (most significant first)
            digits = list(map(int, str(N)))

            # memo dp parameters:
            # pos: current index in digits array
            # is_tight: whether we are bound by the prefix of N
            # has_started: whether we've encountered a non-leading zero so far or not
            # We'll count number of ways to build a valid number from [0..pos].
            from functools import lru_cache

            @lru_cache(None)
            def dp(pos, is_tight, has_started):
                # If we've placed digits for all positions, that's exactly 1 valid number
                if pos == len(digits):
                    return 1

                limit_digit = digits[pos] if is_tight else 9
                total_ways = 0
                for dig in range(0, limit_digit + 1):
                    if dig <= limit:
                        # If we haven't started, we can still place 'dig' = 0
                        # or if we've started, we can place any digit up to limit_digit
                        new_is_tight = (is_tight and (dig == limit_digit))
                        # new_has_started = has_started or dig != 0
                        # For this problem, leading zeros are fine, they don't cause invalid digit usage
                        total_ways += dp(pos + 1, new_is_tight, has_started or dig != 0)
                    else:
                        # Once dig > limit, break because no further digits are valid.
                        break

                return total_ways

            return dp(0, True, False)

        # We use the digit-DP for [0..high] minus [0..low-1]
        res = count_valid_digits_up_to(high, limit)
        if low > 0:
            res -= count_valid_digits_up_to(low - 1, limit)
        return res