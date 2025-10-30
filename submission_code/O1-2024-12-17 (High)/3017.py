class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        """
        We want to count how many numbers x in [low, high] satisfy:
          1) x is divisible by k
          2) x has an equal count of even and odd digits

        Because high can be up to 10^9, iterating directly is not feasible.
        Instead, we use digit-DP (dynamic programming over the decimal digits).

        The core idea:
         - Define a function count_beautiful(n, k) that counts how many integers
           in the range [0, n] meet the "beautiful" criteria.
         - Then our answer is count_beautiful(high, k) - count_beautiful(low - 1, k).

        Digit-DP state explanation:
         We process the decimal representation of n from most significant digit
         to least significant digit. At each position 'pos', we keep track of:
          - 'diff': (#even_digits - #odd_digits) so far (can be negative).
          - 'rem': the current remainder when divided by k.
          - 'leading_zero': whether we have only chosen '0' digits so far
                            (i.e. we haven't started the number yet).
          - 'is_tight': whether we are bound by the prefix of n (if True, the
                        next digit cannot exceed the digit in n at this position).

         At the end (pos == len(digits)), we check if:
          - diff == 0  (equal count of even and odd digits)
          - rem == 0   (the number is divisible by k)
          - leading_zero == False (exclude the "all zero" case which would
            otherwise represent the number 0; also 0 is not "beautiful" because
            it has 1 even digit and 0 odd digits).

        This method counts all valid numbers up to n in O(#digits * range_of_diff * k * 2 * 2).
        For 10^9, #digits <= 10, diff ranges in [-10..10], k <= 20, so it is efficient enough.
        """

        def count_beautiful(n: int, k: int) -> int:
            # If n < 0, there are no valid numbers <= n.
            if n < 0:
                return 0

            # Convert n to a list of digits.
            digits = list(map(int, str(n)))
            length = len(digits)

            # Memo dictionary: key = (pos, diff, rem, leading_zero, is_tight)
            #                  value = count of valid ways
            memo = {}

            def dfs(pos, diff, rem, leading_zero, is_tight):
                # If we've placed all digits, check if conditions are met
                if pos == length:
                    # We want diff = 0, rem = 0, and not leading_zero
                    return 1 if (diff == 0 and rem == 0 and not leading_zero) else 0

                # If we have already computed this state, return it
                key = (pos, diff, rem, leading_zero, is_tight)
                if key in memo:
                    return memo[key]

                limit = digits[pos] if is_tight else 9
                ways = 0
                for d in range(limit + 1):
                    # next_leading_zero: we stay in leading_zero if we choose d=0 and were already in leading_zero
                    next_leading_zero = leading_zero and (d == 0)

                    # If we are still in leading_zero, 'diff' doesn't change.
                    # Once we leave leading_zero (choose a nonzero d), we update diff accordingly.
                    if next_leading_zero:
                        next_diff = diff
                    else:
                        # d is even if d % 2 == 0, odd otherwise
                        if d % 2 == 0:
                            next_diff = diff + 1
                        else:
                            next_diff = diff - 1

                    next_rem = (rem * 10 + d) % k
                    next_is_tight = (is_tight and (d == limit))

                    # Prune impossible 'diff' quickly if we like (optional),
                    # though not strictly needed since diff can only go
                    # as far as -10..10 for up to 10 digits.
                    if abs(next_diff) > length:  
                        # This is just a safe pruning since we won't exceed 10 anyway.
                        continue

                    ways += dfs(pos + 1, next_diff, next_rem, next_leading_zero, next_is_tight)

                memo[key] = ways
                return ways

            return dfs(0, 0, 0, True, True)

        # The result is the count up to 'high' minus the count up to 'low-1'.
        return count_beautiful(high, k) - count_beautiful(low - 1, k)