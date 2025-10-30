from typing import List
import math

# Precompute factorials
# MAXN is up to 100, so we need factorials up to 100.
# Python handles arbitrary precision integers, so no overflow issue with factorial values.
MAXN = 100
factorial = [1] * (MAXN + 1)
for i in range(2, MAXN + 1):
    factorial[i] = factorial[i - 1] * i

class Solution:
    def count_valid(self, o: int, e: int, start_parity: int) -> int:
         """
         Calculates the number of alternating permutations using o odd and e even numbers
         that form a sequence of length o+e starting with start_parity.
         start_parity: 0 for even, 1 for odd.
         
         An alternating permutation of length L = o+e starting with parity P
         (0 for Even, 1 for Odd) requires:
         Number of odd positions = (L + P) // 2
         Number of even positions = (L + 1 - P) // 2
         
         If the number of available odd numbers `o` equals the required odd positions
         AND the number of available even numbers `e` equals the required even positions,
         then we can arrange the `o` odd numbers into `o!` ways and the `e` even numbers
         into `e!` ways independently into their designated slots. The total count is o! * e!.
         Otherwise, it's impossible to form this pattern with these counts, so the count is 0.
         """
         length = o + e
         if length == 0:
             return 1 # Base case: there is 1 way to arrange 0 elements

         # Required number of odd/even positions for this length and starting parity pattern
         o_req = (length + start_parity) // 2
         e_req = (length + 1 - start_parity) // 2

         # Check if the available counts match the required counts for this pattern
         if o == o_req and e == e_req:
             return factorial[o] * factorial[e]
         else:
             return 0

    def permute(self, n: int, k: int) -> List[int]:
        available_odd = [i for i in range(1, n + 1) if i % 2 != 0]
        available_even = [i for i in range(1, n + 1) if i % 2 == 0]

        num_odd_n = len(available_odd)
        num_even_n = len(available_even)

        # Calculate total number of alternating permutations of {1, ..., n}
        # This is the sum of permutations starting with odd and starting with even.
        total_perms = self.count_valid(num_odd_n, num_even_n, 1) # Total starting Odd (OEOE...)
        total_perms += self.count_valid(num_odd_n, num_even_n, 0) # Total starting Even (EOEO...)

        if k > total_perms:
            return []

        result = []

        # Build the permutation element by element
        for i in range(n):
            rem_odd = len(available_odd)
            rem_even = len(available_even)

            # Determine allowed parity for the next element at position i
            # The first element (i=0) can be odd or even.
            # Subsequent elements must have the opposite parity of the previous element.
            allow_odd = (i == 0 or result[-1] % 2 == 0) # Allowed if previous was even or it's the first element
            allow_even = (i == 0 or result[-1] % 2 != 0) # Allowed if previous was odd or it's the first element

            candidates_info = [] # Store (value, is_odd) for available numbers matching allowed parity

            # Collect potential candidates based on allowed parity and availability
            if allow_odd and available_odd:
                # Consider all available odd numbers as candidates for this position.
                # Note: We only need to consider available_odd[0] if we are just counting how many
                # permutations start with *any* odd number. When finding the k-th lexicographical
                # permutation, we must consider all eligible available numbers in increasing order.
                 for val in available_odd:
                     candidates_info.append((val, True))


            if allow_even and available_even:
                 # Consider all available even numbers as candidates for this position
                 for val in available_even:
                      candidates_info.append((val, False))

            # Sort candidates by value to process in lexicographical order
            # This list now contains all available numbers that can legally be placed at position i, sorted.
            candidates_info.sort()

            # Find the correct candidate for position i using k
            found_candidate = False
            for value, is_odd in candidates_info:
                # Calculate the number of complete alternating permutations
                # that can be formed by choosing 'value' at position i,
                # followed by an alternating permutation of the *remaining numbers*
                # starting with the required opposite parity.

                # The remaining numbers will have counts:
                #   rem_odd - 1 odd, rem_even even (if 'value' was odd)
                #   rem_odd odd, rem_even - 1 even (if 'value' was even)
                # The length of the remaining sequence is n - (i + 1) = rem_odd + rem_even - 1.
                # The starting parity for the remaining sequence is the opposite of 'value'.

                if is_odd:
                    # If we choose this odd 'value', the remaining rem_odd-1 odd and rem_even even numbers
                    # must form an alternating sequence starting with even (0).
                    count = self.count_valid(rem_odd - 1, rem_even, 0)
                else: # is_even
                    # If we choose this even 'value', the remaining rem_odd odd and rem_even-1 even numbers
                    # must form an alternating sequence starting with odd (1).
                    count = self.count_valid(rem_odd, rem_even - 1, 1)

                # If k falls within the range covered by permutations starting with 'value' at position i
                if k <= count:
                    # This 'value' is the correct number for the current position i.
                    result.append(value)
                    # Remove the chosen value from the available list
                    if is_odd:
                        available_odd.remove(value) # O(N) operation on list
                    else:
                        available_even.remove(value) # O(N) operation on list
                    found_candidate = True
                    break # Move to the next position i+1
                else:
                    # If k is larger, these 'count' permutations starting with 'value' are skipped.
                    k -= count

            # If we exit the loop without finding a candidate, it implies k was initially
            # too large for any valid permutation, or there is a logic error.
            # The initial check `k > total_perms` should ideally prevent this.
            # An assertion could be added here for debugging if needed: assert found_candidate

        return result