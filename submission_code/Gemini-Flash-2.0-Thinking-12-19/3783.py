from typing import List
import sys
# Increase recursion depth for memoization. Max depth is n.
# For n=100, recursion depth can be around 100. Default might be too low.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 100 + 10)) # Add buffer

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        odd_count = (n + 1) // 2
        even_count = n // 2

        # Memoization table for count_alt(o, e, start_odd)
        # key: (o, e, start_odd), value: count
        memo = {}

        def count_alt(o, e, start_odd):
            """
            Counts the number of alternating permutations using `o` abstract odd numbers
            and `e` abstract even numbers, starting with an odd number if `start_odd` is True,
            or an even number if `start_odd` is False.
            """
            if (o, e, start_odd) in memo:
                return memo[(o, e, start_odd)]

            # Base case: empty sequence (0 numbers used)
            # Represents the successful completion of a sub-permutation sequence.
            # There is 1 way to arrange 0 numbers.
            if o == 0 and e == 0:
                return 1

            count = 0
            # If the sequence being counted must start with an odd type
            if start_odd:
                if o > 0:
                    # Choose one of the 'o' abstract odd slots for the first position.
                    # The remaining o-1 odd slots and e even slots must form a sequence starting with even.
                    count = o * count_alt(o - 1, e, False)
            # If the sequence being counted must start with an even type
            else:
                if e > 0:
                    # Choose one of the 'e' abstract even slots for the first position.
                    # The remaining o odd slots and e-1 even slots must form a sequence starting with odd.
                    count = e * count_alt(o, e - 1, True)

            memo[(o, e, start_odd)] = count
            return count

        # Check if k is valid against the total number of alternating permutations
        # The total number of alternating permutations of [1...n] is
        # (Those starting with odd) + (Those starting with even)
        total_perms_start_odd = count_alt(odd_count, even_count, True)
        total_perms_start_even = count_alt(odd_count, even_count, False)
        total_alt_perms = total_perms_start_odd + total_perms_start_even

        # k is 1-indexed. If k > total valid permutations, return empty list.
        if k > total_alt_perms:
            return []

        result = []
        k_rem = k # Use k as the remaining index (1-indexed)
        used = [False] * (n + 1) # used[i] is True if number i is used

        current_odd_count = odd_count
        current_even_count = even_count

        # Build the permutation element by element
        for _ in range(n):
            needed_parity = -1 # 0 for even, 1 for odd

            # Determine the required parity for the next element based on the last element placed
            if len(result) > 0:
                last_num = result[-1]
                # If last was odd (1), needed is even (0). If last was even (0), needed is odd (1).
                needed_parity = 1 - (last_num % 2)

            # Iterate through numbers 1 to n in increasing order to find the next digit lexicographically
            for num in range(1, n + 1):
                if not used[num]:
                    num_parity = num % 2 # 0 if even, 1 if odd

                    # Check if 'num' is a valid candidate for the current position based on alternating property
                    is_valid_candidate = False
                    if len(result) == 0:
                        # For the first element, any number is a valid candidate to start either the OEOE... pattern or the EOEO... pattern.
                        is_valid_candidate = True
                    else:
                        # For subsequent elements, the parity must alternate with the previous element.
                        if num_parity == needed_parity:
                            is_valid_candidate = True

                    # If the number 'num' is a valid candidate for this position
                    if is_valid_candidate:
                        # Calculate how many alternating permutations can be formed using the *remaining* numbers,
                        # assuming 'num' is placed next, and then continuing with the required alternating parity.

                        remaining_odds_count = current_odd_count - (1 if num_parity == 1 else 0)
                        remaining_evens_count = current_even_count - (1 if num_parity == 0 else 0)

                        # After placing 'num' (with num_parity), the next element must have the opposite parity (1 - num_parity).
                        # The suffix sequence must start with this opposite parity.
                        # The `start_odd` parameter in `count_alt` is True if the suffix should start with odd (parity 1), False if with even (parity 0).
                        # If `num_parity` is 1 (odd), `1 - num_parity` is 0 (even), suffix must start with even, `start_odd` is False.
                        # If `num_parity` is 0 (even), `1 - num_parity` is 1 (odd), suffix must start with odd, `start_odd` is True.
                        # This mapping is `(num_parity == 0)`.
                        required_start_parity_for_suffix = (num_parity == 0) # True if suffix needs to start with odd (1), False if suffix needs to start with even (0)


                        # Count the number of alternating permutations of the remaining numbers, starting with the required parity for the suffix.
                        num_suffixes = count_alt(remaining_odds_count, remaining_evens_count, required_start_parity_for_suffix)

                        # Check if the k-th permutation (among the remaining ones) falls within the block
                        # of permutations that start with 'num' at the current position.
                        # Since k_rem is 1-indexed, if k_rem is <= num_suffixes, the target permutation is in this block.
                        if k_rem <= num_suffixes:
                            # 'num' is the correct digit for the current position.
                            result.append(num)
                            used[num] = True # Mark number as used
                            # Update counts of available odds/evens
                            current_odd_count = remaining_odds_count
                            current_even_count = remaining_evens_count
                            # k_rem does not change, as we found the k_rem-th permutation within this block.
                            break # Move to the next position in the permutation

                        else:
                            # 'num' is not the correct digit. Skip all permutations starting with 'num' at this position
                            # by subtracting the count from k_rem.
                            k_rem -= num_suffixes

        return result