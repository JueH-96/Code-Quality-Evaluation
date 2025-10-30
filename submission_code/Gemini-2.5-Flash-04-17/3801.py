import functools

# Max exponents required for primes 2, 3, 5, 7 for sums up to 81
CAP_EXP2 = 7 # Max needed is 6 (for 64 = 2^6)
CAP_EXP3 = 5 # Max needed is 4 (for 81 = 3^4)
CAP_EXP5 = 3 # Max needed is 2 (for 25 = 5^2)
CAP_EXP7 = 3 # Max needed is 2 (for 49 = 7^2)

# Precompute prime factorizations for sums 1 to 81
# For S, store {p: a} where p^a divides S
# Also store the remainder factor > 7
SUM_FACTORIZATION = {}
for s in range(1, 82): # Possible sums of digits for numbers < 10^9 range from 1 to 81.
    factors = {}
    temp_s = s
    # Check divisibility by primes <= 7
    for p in [2, 3, 5, 7]:
        if temp_s == 0: break # Should not happen for s>=1
        while temp_s > 0 and temp_s % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp_s //= p
    # Store the remaining factor (product of primes > 7)
    if temp_s > 1:
        factors['rem'] = temp_s
    SUM_FACTORIZATION[s] = factors

# Precompute exponents of primes 2, 3, 5, 7 for digits 1 through 9.
# Digit 0 contributes to the product being 0, handled by 'has_zero' flag.
DIGIT_EXPONENTS = {}
for d in range(1, 10):
    factors = {2: 0, 3: 0, 5: 0, 7: 0}
    temp_d = d
    for p in [2, 3, 5, 7]:
        while temp_d > 0 and temp_d % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp_d //= p
    DIGIT_EXPONENTS[d] = factors


class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        # The count of beautiful numbers in [l, r] is count(r) - count(l-1)
        return self.count_beautiful_up_to(r) - self.count_beautiful_up_to(l - 1)

    def count_beautiful_up_to(self, N):
        """
        Counts the number of beautiful positive integers x such that 1 <= x <= N.
        A positive integer is beautiful if the product of its digits is divisible by the sum of its digits.
        Uses digit DP.
        """
        if N < 1:
            return 0 # No positive integers <= 0

        n_str = str(N)
        D = len(n_str) # Number of digits in N

        # Memoization table to store results of dp states.
        # Key: (idx, current_sum, exp2, exp3, exp5, exp7, has_zero, is_strictly_leading, tight)
        # idx: Current digit position (0 to D-1).
        # current_sum: Sum of digits placed so far (for the actual number, excluding leading zeros).
        # expX: Accumulated exponent of prime X (2, 3, 5, 7) in the product of non-zero digits placed so far. Capped.
        # has_zero: Boolean, True if a digit 0 has been placed in the actual number.
        # is_strictly_leading: Boolean, True if all digits placed so far are leading zeros.
        # tight: Boolean, True if we are restricted by the digits of N.
        memo = {}

        def dp(idx, current_sum, exp2, exp3, exp5, exp7, has_zero, is_strictly_leading, tight):
            """
            Recursive DP function to count beautiful numbers.
            Counts numbers that can be formed using digits from index `idx` to D-1,
            given the constraints and properties accumulated from indices 0 to idx-1.
            """
            # Memoization lookup
            state_key = (idx, current_sum, exp2, exp3, exp5, exp7, has_zero, is_strictly_leading, tight)
            if state_key in memo:
                return memo[state_key]

            # Base case: All digits processed
            if idx == D:
                # If is_strictly_leading is true, it means we only placed leading zeros,
                # resulting in the number 0, which is not a positive beautiful number.
                if is_strictly_leading:
                    return 0

                # We have formed a positive number. Check beautiful number conditions.
                # The current_sum is the final sum of digits.
                S = current_sum

                # S must be > 0 for a positive number. This is guaranteed because
                # `!is_strictly_leading` means at least one non-zero digit was placed.

                # Check product divisibility by S.
                s_factors = SUM_FACTORIZATION[S] # S is guaranteed >= 1 here

                s_rem = s_factors.get('rem', 1) # Remainder after dividing by 2, 3, 5, 7 factors

                # Case 1: S has prime factors > 7 (s_rem > 1)
                # The product of digits (which only has factors 2, 3, 5, 7, or is 0)
                # can only be divisible by a number with factors > 7 if the product is 0.
                # Product is 0 iff `has_zero` is True.
                if s_rem > 1:
                    return 1 if has_zero else 0

                # Case 2: S only has prime factors 2, 3, 5, 7 (s_rem == 1)
                # If a zero was used, product is 0, divisible by any S > 0.
                if has_zero:
                    return 1

                # If no zero was used, the product is the product of non-zero digits.
                # Check if accumulated exponents of 2, 3, 5, 7 in the product
                # are sufficient to satisfy the divisibility requirements from S.
                if exp2 >= s_factors.get(2, 0) and \
                   exp3 >= s_factors.get(3, 0) and \
                   exp5 >= s_factors.get(5, 0) and \
                   exp7 >= s_factors.get(7, 0):
                   return 1
                else:
                   return 0

            # Recursive step
            ans = 0
            # The upper bound for the current digit depends on the 'tight' constraint.
            upper_bound = int(n_str[idx]) if tight else 9

            # Iterate through possible digits for the current position
            for digit in range(upper_bound + 1):
                # If we are currently in the leading zeros phase and choose 0,
                # we remain in the leading zeros phase. The state variables (sum, exp, has_zero)
                # reflecting the actual number properties do not change conceptually yet.
                if is_strictly_leading and digit == 0:
                    # Recurse: move to the next index, keep state variables as initial, remain strictly leading.
                    # Update tight constraint: true only if we chose 0 and the upper_bound was 0 (implies n_str[idx] was '0').
                    # Since upper_bound is int(n_str[idx]), digit==upper_bound check is correct.
                    ans += dp(idx + 1, 0, 0, 0, 0, 0, False, True, tight and (digit == upper_bound))
                else:
                    # This is the first non-zero digit OR we are past the leading zeros phase.
                    # The state variables will now track the properties of the actual number being built.
                    # The next state will not be strictly leading.

                    # Calculate the new state variables based on the chosen digit.
                    new_current_sum = current_sum + digit
                    # has_zero becomes true if it was already true, or if the current digit is 0.
                    new_has_zero = has_zero or (digit == 0)

                    # Update exponents for prime factors 2, 3, 5, 7 based on the current digit.
                    # Digit 0 does not contribute factors to the non-zero product part.
                    new_exp2 = exp2
                    new_exp3 = exp3
                    new_exp5 = exp5
                    new_exp7 = exp7

                    if digit > 0:
                        d_exps = DIGIT_EXPONENTS[digit]
                        new_exp2 = min(CAP_EXP2, exp2 + d_exps[2])
                        new_exp3 = min(CAP_EXP3, exp3 + d_exps[3])
                        new_exp5 = min(CAP_EXP5, exp5 + d_exps[5])
                        new_exp7 = min(CAP_EXP7, exp7 + d_exps[7])
                    # Note: if digit is 0 and not is_strictly_leading, new_has_zero becomes true,
                    # but exp variables do not change as they track non-zero product factors.

                    # Pruning: If adding this digit makes the current sum exceed the maximum possible sum (81),
                    # this branch cannot lead to a beautiful number < 10^9.
                    # The maximum possible sum for a number < 10^9 is 81 (for 999,999,999).
                    if new_current_sum > 81:
                         # This pruning is valid because digits are tried in increasing order (0..upper_bound).
                         # If `current_sum + digit > 81`, then `current_sum + (digit + 1)` will also be > 81,
                         # and any subsequent digit will also lead to a sum > 81.
                         break

                    # Recurse: move to the next index, pass updated state variables, next state is not strictly leading.
                    # Update tight constraint: true only if we chose the upper_bound digit and the current state was tight.
                    ans += dp(idx + 1, new_current_sum, new_exp2, new_exp3, new_exp5, new_exp7, new_has_zero, False, tight and (digit == upper_bound))

            # Store the computed result in the memoization table
            memo[state_key] = ans
            return ans

        # Initial call to the DP function to count beautiful numbers up to N.
        # Start at index 0, with initial sum 0, initial exponents 0, no zero seen yet,
        # in the strictly leading phase, and restricted by the digits of N (tight=True).
        return dp(0, 0, 0, 0, 0, 0, False, True, True)