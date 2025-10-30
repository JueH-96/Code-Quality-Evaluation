from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # We need to find pairs (i, j) such that i < j and (hours[i] + hours[j]) % 24 == 0.
        # This is equivalent to checking the remainders when hours[k] is divided by 24.
        # Let rem_i = hours[i] % 24 and rem_j = hours[j] % 24.
        # The condition becomes (rem_i + rem_j) % 24 == 0.

        # We use an array to store the frequency of each remainder (0 to 23).
        # The size of the array is 24, as remainders are 0, 1, ..., 23.
        remainder_counts = [0] * 24

        # First pass: Populate the remainder_counts array
        # For each hour value, calculate its remainder when divided by 24 and
        # increment the corresponding count in our remainder_counts array.
        for h in hours:
            remainder_counts[h % 24] += 1

        pair_count = 0

        # Case 1: Pairs where both numbers have a remainder of 0 when divided by 24.
        # If there are 'n' numbers with remainder 0, they can form n * (n - 1) / 2 pairs.
        # Example: [24, 48, 72] -> remainder_counts[0] = 3. Pairs: (24,48), (24,72), (48,72). Total 3 * 2 / 2 = 3.
        n0 = remainder_counts[0]
        pair_count += n0 * (n0 - 1) // 2

        # Case 2: Pairs where both numbers have a remainder of 12 when divided by 24.
        # 12 + 12 = 24, which is a multiple of 24.
        # Similar to remainder 0, if 'n' numbers have remainder 12, they form n * (n - 1) / 2 pairs.
        n12 = remainder_counts[12]
        pair_count += n12 * (n12 - 1) // 2

        # Case 3: Pairs where remainders sum to 24 (excluding 0 and 12, already handled).
        # We iterate through remainders from 1 to 11 (inclusive).
        # For each remainder `i`, we look for its complement `24 - i`.
        # Example: if i = 1, complement is 23. (1 + 23 = 24).
        # The number of pairs between elements with remainder `i` and elements with remainder `24 - i`
        # is `remainder_counts[i] * remainder_counts[24 - i]`.
        # We only iterate up to 11 because iterating up to 23 would double-count.
        # (e.g., when i=1, we count pairs with 23; when i=23, we'd count pairs with 1 again).
        for i in range(1, 12): # i will range from 1, 2, ..., 11
            pair_count += remainder_counts[i] * remainder_counts[24 - i]
            
        return pair_count