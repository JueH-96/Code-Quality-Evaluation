from typing import List, Dict, Tuple

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Use a dictionary for memoization. Key: (length, avail_o, avail_e, start_odd)
        # Value: Number of alternating permutations that can be formed
        # using avail_o odd abstract numbers and avail_e even abstract numbers,
        # starting with the specified parity.
        memo: Dict[Tuple[int, int, int, bool], int] = {}

        def count(length: int, avail_o: int, avail_e: int, start_odd: bool) -> int:
            """
            Counts alternating permutations of length 'length' using a pool of
            'avail_o' abstract odd numbers and 'avail_e' abstract even numbers.
            'start_odd': True if the first element must be odd, False if even.
            """
            # Base case: An empty permutation (length 0) can be formed in 1 way.
            if length == 0:
                return 1

            # Invalid states: Cannot form a permutation if we need elements but have none left,
            # or if the number of available items doesn't match the required length.
            # The check `avail_o + avail_e != length` is valid here because this function
            # counts permutations of *exactly* `avail_o` odd and `avail_e` even items,
            # implying the total number of items available is `avail_o + avail_e`.
            if avail_o < 0 or avail_e < 0 or avail_o + avail_e != length:
                 return 0

            # Check memoization table to avoid redundant calculations
            if (length, avail_o, avail_e, start_odd) in memo:
                return memo[(length, avail_o, avail_e, start_odd)]

            result = 0
            if start_odd: # The first element of this subproblem must be odd
                if avail_o > 0:
                    # If we choose one of the 'avail_o' odd numbers, the remaining
                    # length is length-1. We have avail_o-1 odd and avail_e even
                    # abstract numbers left. The next element must be even.
                    result = avail_o * count(length - 1, avail_o - 1, avail_e, False)
            else: # The first element of this subproblem must be even
                if avail_e > 0:
                    # If we choose one of the 'avail_e' even numbers, the remaining
                    # length is length-1. We have avail_o odd and avail_e-1 even
                    # abstract numbers left. The next element must be odd.
                    result = avail_e * count(length - 1, avail_o, avail_e - 1, True)

            # Store the calculated result in the memoization table
            memo[(length, avail_o, avail_e, start_odd)] = result
            return result

        # Calculate the total number of odd and even integers in the range [1, n]
        num_odd = (n + 1) // 2 # Equivalent to ceil(n/2)
        num_even = n // 2     # Equivalent to floor(n/2)

        # Calculate the total number of valid alternating permutations of {1, ..., n}.
        # A permutation can start with either an odd or an even number.
        # We use the count function with the full set of numbers {1..n}.
        total_o_start = count(n, num_odd, num_even, True)
        total_e_start = count(n, num_odd, num_even, False)
        total_permutations = total_o_start + total_e_start

        # If the requested k is greater than the total number of valid permutations,
        # it means the k-th permutation does not exist. Return an empty list.
        # The problem uses 1-indexed k.
        if k > total_permutations:
            return []

        # Initialize data structures for building the k-th permutation element by element
        permutation = []
        # `used` array tracks which numbers from 1 to n have been placed in the permutation.
        # We use 1-based indexing for convenience, so the size is n + 1.
        used = [False] * (n + 1)
        # `o_used_count` and `e_used_count` track how many odd and even numbers
        # from the original set {1..n} have been used so far in the permutation.
        o_used_count = 0
        e_used_count = 0

        # Build the k-th permutation element by element from left to right (position i = 0 to n-1)
        for i in range(n):
            # Determine the required parity for the number at the current position 'i'.
            # For the first position (i=0), there's no constraint from a previous element,
            # so any parity can potentially start the permutation.
            # For subsequent positions (i>0), the parity must be opposite to the previous element placed at position i-1.
            required_current_parity = None # Initialize as None for the first position
            if i > 0:
                prev_element = permutation[-1]
                prev_parity = prev_element % 2 # 1 if odd, 0 if even
                required_current_parity = 1 - prev_parity # Required parity is the opposite

            # Iterate through possible values (1 to n) in increasing order.
            # This iteration order ensures we find the lexicographically smallest valid number first.
            for val in range(1, n + 1):
                # Skip the number if it has already been used in the permutation.
                if used[val]:
                    continue

                current_parity = val % 2 # 1 if val is odd, 0 if val is even

                # Check if placing 'val' at the current position 'i' satisfies the alternating property.
                is_valid_candidate = True
                if required_current_parity is not None and current_parity != required_current_parity:
                    is_valid_candidate = False # 'val' has the same parity as the previous element

                # If 'val' is a valid number to place at the current position 'i'
                if is_valid_candidate:
                    # Calculate the number of valid alternating permutations that can be formed
                    # by choosing 'val' as the element at position 'i' and completing the remaining slots.
                    # The remaining part of the permutation needs to fill n - (i + 1) positions.
                    remaining_length = n - (i + 1)
                    
                    # Count the number of odd and even integers *from the original set {1..n}*
                    # that are still available after hypothetically choosing 'val' for position 'i'.
                    rem_avail_o_for_count = (num_odd - o_used_count) - (1 if current_parity == 1 else 0)
                    rem_avail_e_for_count = (num_even - e_used_count) - (1 if current_parity == 0 else 0)

                    # Determine the required parity for the element at the very next position (i+1),
                    # which will be the first element of the permutation suffix we are counting.
                    req_next_parity_is_odd = (1 - current_parity == 1) # True if next must be odd

                    # Use the 'count' function to find how many ways the remaining positions
                    # can be filled to form a valid alternating sequence, given the available
                    # remaining numbers and the required starting parity for the suffix.
                    num_ways = count(remaining_length, rem_avail_o_for_count, rem_avail_e_for_count, req_next_parity_is_odd)

                    # If k is less than or equal to the number of permutations that start with
                    # the current prefix followed by 'val', it means the k-th permutation
                    # starts with this prefix. Therefore, 'val' is the correct element
                    # for the current position 'i'.
                    if k <= num_ways:
                        # Add 'val' to the permutation being built.
                        permutation.append(val)
                        # Mark 'val' as used.
                        used[val] = True
                        # Update the counts of used odd/even numbers from the original set.
                        if current_parity == 1:
                            o_used_count += 1
                        else:
                            e_used_count += 1
                        # Since we found the correct element for position 'i', break out of
                        # the inner loop and proceed to find the element for the next position (i+1).
                        break
                    else:
                        # 'val' is not the correct element for position 'i' in the k-th permutation.
                        # The k-th permutation must be one of the permutations that start with a
                        # lexicographically larger valid number than 'val' at position 'i'.
                        # Subtract the number of permutations that would start with 'val' here from k.
                        k -= num_ways
                        # Continue the inner loop to check the next possible value for position 'i'.

        # After the loop completes for all n positions, we have successfully constructed the k-th permutation.
        return permutation