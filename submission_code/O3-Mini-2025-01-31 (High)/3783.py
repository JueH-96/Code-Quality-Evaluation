from typing import List
from math import factorial

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Precompute factorials up to 50.
        # (Since in an alternating permutation the largest group size is ≤ 50 when n ≤ 100.)
        max_val = 50  
        fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = fact[i - 1] * i

        # Count how many odds and evens are in 1..n.
        odd_total = (n + 1) // 2
        even_total = n // 2

        # Total number of valid alternating permutations (global order) 
        # comes from a case‐analysis on the first element.
        # For n odd, the only valid first element is odd.
        # For n even, both can be chosen.
        if n % 2 == 1:
            total_count = fact[odd_total] * fact[even_total]
        else:
            # When n is even we have two disjoint groups:
            #   Pattern “odd-first”: first element odd, then even, odd, even,...
            #   Pattern “even-first”: first element even, then odd, even, odd,...
            # For either pattern, the count is (group_size)!*(group_size)! 
            # and the global total is the sum for both patterns.
            total_count = 2 * fact[even_total] * fact[even_total]

        if k > total_count:
            return []

        result = []
        # Prepare lists for available odd and even numbers.
        available_odds = [x for x in range(1, n + 1) if x % 2 == 1]
        available_evens = [x for x in range(1, n + 1) if x % 2 == 0]

        # At the very first position we are free to choose any number
        # (with the caveat that if n is odd, only odd numbers can start a valid alternating permutation).
        # We iterate in increasing order (this ensures lexicographical order).
        pattern = None  # This will be either "odd-first" or "even-first" (i.e. the fixed parity pattern)
        for x in range(1, n + 1):
            if n % 2 == 1 and x % 2 == 0:
                # For odd n, a valid alternating permutation must start with an odd number.
                continue

            if x % 2 == 1:
                # If we choose an odd number first, the fixed pattern will be:
                #   odd-index positions must be odd, even-index positions must be even.
                candidate_count = fact[len(available_odds) - 1] * fact[len(available_evens)]
                candidate_pattern = "odd-first"
            else:
                # For n even a first element can be even.
                # Then the pattern is forced to:
                #   odd-index positions are even and even-index positions are odd.
                candidate_count = fact[len(available_evens) - 1] * fact[len(available_odds)]
                candidate_pattern = "even-first"

            if k > candidate_count:
                k -= candidate_count
            else:
                result.append(x)
                if candidate_pattern == "odd-first":
                    available_odds.remove(x)
                else:
                    available_evens.remove(x)
                pattern = candidate_pattern
                break

        # Now the pattern is fixed.
        # For "odd-first": positions 1,3,5,... must be odd and positions 2,4,6,... even.
        # For "even-first": positions 1,3,5,... must be even and positions 2,4,6,... odd.
        # We now decide the rest of the permutation one position at a time.
        for pos in range(2, n + 1):
            if pattern == "odd-first":
                req = "odd" if pos % 2 == 1 else "even"
            else:  # pattern == "even-first"
                req = "even" if pos % 2 == 1 else "odd"

            if req == "odd":
                candidate_list = available_odds
            else:
                candidate_list = available_evens

            chosen = None
            # The candidate list is already sorted.
            for c in candidate_list:
                # At any state the remaining valid completions 
                # come from independently permuting the remaining numbers in each parity group.
                # So if we are about to choose one candidate from the required group,
                # the number of completions after that choice would be:
                #   ( (size of that group - 1)! * (size of other group)! ).
                if req == "odd":
                    c_count = fact[len(available_odds) - 1] * fact[len(available_evens)]
                else:
                    c_count = fact[len(available_evens) - 1] * fact[len(available_odds)]
                if k > c_count:
                    k -= c_count
                else:
                    chosen = c
                    result.append(c)
                    if req == "odd":
                        available_odds.remove(c)
                    else:
                        available_evens.remove(c)
                    break

            if chosen is None:
                # This should not happen if k is ≤ total_count.
                return []

        return result