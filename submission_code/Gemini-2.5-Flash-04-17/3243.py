import functools

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:

        def count_powerful_up_to(N: int, limit: int, s: str) -> int:
            N_str = str(N)
            n = len(N_str)
            m = len(s)

            # If the smallest possible powerful number (int(s)) is greater than N,
            # no powerful number can be less than or equal to N.
            # Compare s and N_str as strings for large numbers.
            # If m > n, s is numerically larger than any number with n digits.
            # If m == n, compare s and N_str lexicographically.
            if m > n or (m == n and s > N_str):
                return 0

            total_count = 0

            # Case 1: Powerful numbers with length L, where m <= L < n
            # These numbers are always <= N because they have fewer digits than N.
            for L in range(m, n):
                prefix_len = L - m
                if prefix_len == 0:
                    # Length L = m, the number is s itself.
                    # Since m < n, int(s) < any number of length n, so int(s) <= N.
                    # Digits of s are <= limit by constraint. s is positive.
                    total_count += 1
                else:
                    # Length L > m. The number is prefix (length prefix_len > 0) + s.
                    # The first digit of the prefix (at number index 0) cannot be 0.
                    # It can be 1 to limit (limit choices).
                    # The remaining prefix_len - 1 digits (at number indices 1 to prefix_len - 1)
                    # can be 0 to limit (limit + 1 choices each).
                    num_prefixes = limit * (limit + 1)**(prefix_len - 1)
                    total_count += num_prefixes

            # Case 2: Powerful numbers with length L = n, and <= N
            # These numbers are of the form prefix (length n-m) + s (length m)
            # and must be numerically <= N_str.
            # Use digit DP to count these.
            # state: (index, is_tight)
            # index: current digit position we are filling (0 to n-1)
            # is_tight: boolean, true if we are restricted by the digits of N_str

            @functools.lru_cache(None)
            def count_len_n(index: int, is_tight: bool) -> int:
                # Base case: successfully filled all n digits
                if index == n:
                    return 1

                # If the current position is within the suffix part (indices n-m to n-1)
                # This block is reached only if n >= m.
                # If n == m, n-m is 0, so indices 0 to m-1 are suffix.
                # If n > m, n-m > 0, suffix starts from index n-m.
                if index >= n - m:
                    # The digit at this position is fixed by s
                    # s_index = index - (n - m)
                    req_digit = int(s[index - (n - m)])
                    upper_bound_N_digit = int(N_str[index])

                    # The digit from s must be <= limit (guaranteed by constraints)
                    # if req_digit > limit: return 0 # This check is redundant due to constraints

                    # If we are in a tight state, the s digit must be <= the corresponding digit in N_str
                    if is_tight and req_digit > upper_bound_N_digit:
                        return 0 # The number being built is already greater than N_str

                    # Move to the next position
                    new_is_tight = is_tight and (req_digit == upper_bound_N_digit)
                    return count_len_n(index + 1, new_is_tight)
                else:
                    # If the current position is within the prefix part (indices 0 to n-m-1)
                    # This block is reached only if n > m (so n-m > 0).
                    upper_bound_N_digit = int(N_str[index])
                    # The maximum digit we can choose is min(limit, upper_bound_N_digit if is_tight else 9)
                    upper_digit_choice = min(limit, upper_bound_N_digit if is_tight else 9)

                    # The minimum digit we can choose is 0.
                    # Special case: The very first digit (at index 0) cannot be 0 if n > m.
                    # If n == m, the prefix length is 0, this `else` block is skipped entirely.
                    # If n > m, the first digit of the prefix is at index 0. It cannot be 0.
                    lower_digit_choice = 0
                    if index == 0 and n > m:
                        lower_digit_choice = 1 # First digit of a number of length n > m cannot be 0

                    res = 0
                    # Iterate through possible digits for the current position
                    for digit in range(lower_digit_choice, upper_digit_choice + 1):
                        new_is_tight = is_tight and (digit == upper_bound_N_digit)
                        res += count_len_n(index + 1, new_is_tight)
                    return res

            # Add count for powerful numbers of length n that are <= N
            # Call DP only if n >= m (already handled by initial check).
            # DP counts length-n numbers. If n == m, the DP counts if s <= N_str.
            total_count += count_len_n(0, True)

            return total_count


        # The total number of powerful integers in [start..finish]
        # is the number powerful integers <= finish minus the number powerful integers < start.
        return count_powerful_up_to(finish, limit, s) - count_powerful_up_to(start - 1, limit, s)