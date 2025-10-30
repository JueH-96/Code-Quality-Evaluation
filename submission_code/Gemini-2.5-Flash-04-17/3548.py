import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        total_good_count = 0

        # Precompute factorials for permutation counting
        # Need factorials up to n
        factorials = [1] * (n + 1)
        for i in range(2, n + 1):
            factorials[i] = factorials[i-1] * i

        def permutations_no_leading_zero(counts, n):
            # counts: list of digit counts [c0, c1, ..., c9]
            # n: total number of digits (sum of counts)

            if n == 0: # Should not happen for problem constraints n >= 1
                return 1

            denominator = 1
            for count in counts:
                # Use precomputed factorials
                denominator *= factorials[count]

            # Total permutations = n! / (c0! c1! ... c9!)
            # Use integer division //
            total_perms = factorials[n] // denominator

            # Valid permutations (no leading zero) = Total * (n - c0) / n
            # This formula works correctly even if c0 is 0 or > 0
            # Use integer division //
            valid_perms = total_perms * (n - counts[0]) // n
            return valid_perms

        # Iterate over the first half digits
        # The number of first half digits to choose is m = (n + 1) // 2
        m = (n + 1) // 2

        # Precompute powers of 10 modulo k
        powers_of_10_mod_k = [1] * n
        for i in range(1, n):
            powers_of_10_mod_k[i] = (powers_of_10_mod_k[i-1] * 10) % k

        def generate_first_half(index, current_half_digits):
            nonlocal total_good_count

            if index == m:
                # We have chosen the first half digits (h_0, h_1, ..., h_{m-1})
                # current_half_digits is a tuple of length m

                # 1. Construct the multiset counts
                counts = [0] * 10
                # Add pairs for digits h_0 to h_{n//2 - 1}
                for i in range(n // 2):
                     counts[current_half_digits[i]] += 2

                # Add middle digit if n is odd (digit h_{n//2})
                if n % 2 == 1:
                     counts[current_half_digits[n // 2]] += 1

                # 2. Check if the corresponding palindrome is divisible by k
                # Palindrome string s_0 s_1 ... s_{n-1}
                # Value P = sum_{p=0}^{n-1} s_p * 10^(n-1-p)
                # Check P % k == 0

                p_mod_k = 0
                for p in range(n):
                    # Get digit s_p based on palindrome structure and current_half_digits
                    digit_at_pos_p = -1 # This will be assigned

                    if p < n // 2:
                        # First half positions [0, n//2 - 1] map to current_half_digits[p]
                        digit_at_pos_p = current_half_digits[p]
                    elif n % 2 == 1 and p == n // 2:
                        # Middle position for odd n maps to current_half_digits[n // 2]
                        digit_at_pos_p = current_half_digits[n // 2]
                    else: # p > n // 2 or (n % 2 == 0 and p == n // 2) covers the second half
                        # Second half positions map to the mirrored index in the first half/middle
                        # The mirror position is n-1-p
                        digit_at_pos_p = current_half_digits[n - 1 - p]

                    # Contribution of this digit to P mod k
                    # The power of 10 for position p (0-indexed) is n-1-p
                    term = (digit_at_pos_p % k) * (powers_of_10_mod_k[n - 1 - p]) % k
                    p_mod_k = (p_mod_k + term) % k

                if p_mod_k == 0:
                    # This multiset of digits can form a k-palindromic number.
                    # Now count distinct n-digit numbers that can be formed
                    # using these digits without leading zero.
                    count_for_this_multiset = permutations_no_leading_zero(counts, n)
                    total_good_count += count_for_this_multiset

                return # Stop recursion branch

            # Recursive step: choose the digit for the current index
            # index 0 corresponds to h_0 (d1), must be 1-9
            # index > 0 corresponds to h_i (di+1), can be 0-9
            start_digit = 0
            if index == 0:
                start_digit = 1

            for digit in range(start_digit, 10):
                # Pass the chosen digit by appending to the tuple
                generate_first_half(index + 1, current_half_digits + (digit,))

        # Start generating first half digits from index 0 with an empty tuple
        generate_first_half(0, ())

        return total_good_count