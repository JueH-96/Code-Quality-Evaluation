from collections import Counter
from typing import List

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the frequency of each character across all words.
        char_counts = Counter()
        for word in words:
            char_counts.update(word)

        # Calculate the initial number of character pairs and singles available globally.
        # 'available_pairs': total pairs of identical characters we can form (e.g., 'aa', 'bb'). These are primarily used for the symmetric sides of palindromes.
        # 'available_singles': total single characters left over after forming initial pairs (e.g., 'c' from a count of 1 or 3). These can be used as centers of odd-length palindromes.
        available_pairs = 0
        available_singles = 0
        for count in char_counts.values():
            available_pairs += count // 2
            available_singles += count % 2

        # Get the lengths of all words and sort them in ascending order.
        # It's optimal to try and form palindromes for the shortest words first, as they require the fewest pairs.
        lengths = [len(word) for word in words]
        lengths.sort()

        # Iterate through the sorted word lengths and greedily try to form palindromes.
        palindromes_made = 0
        for L in lengths:
            required_pairs = L // 2
            required_singles = L % 2 # This is 0 for even lengths, 1 for odd lengths.

            # To form a palindrome of length L, we need `required_pairs` for the symmetric sides
            # and `required_singles` for the center(s).

            # Check if we have enough pairs from the 'available_pairs' pool for the symmetric sides.
            # The 'available_pairs' pool is the primary source for `required_pairs`.
            # If we don't even have enough pairs for the symmetric sides, we cannot form this palindrome.
            if available_pairs >= required_pairs:
                # We have enough pairs for the symmetric sides.
                # Calculate the total pool of single characters available *after* hypothetically
                # using `required_pairs` for the symmetric sides. This pool consists of the
                # original available singles plus any singles obtained by breaking the remaining pairs.
                total_singles_capacity_after_pair_use = available_singles + 2 * (available_pairs - required_pairs)

                # Check if this total singles capacity is sufficient for the required single(s) for the center:
                if total_singles_capacity_after_pair_use >= required_singles:
                    # We have enough resources (pairs for sides + capacity for center) to form this palindrome.

                    # Consume the resources:
                    # 1. Use the required pairs for the symmetric sides.
                    available_pairs -= required_pairs

                    # 2. Use the required singles for the center(s).
                    # These are consumed from the 'available_singles' pool first.
                    singles_to_take_from_available_singles = min(required_singles, available_singles)
                    available_singles -= singles_to_take_from_available_singles

                    # Any remaining required singles must be obtained by breaking pairs.
                    singles_remaining_to_take = required_singles - singles_to_take_from_available_singles

                    # Each pair broken yields 2 singles. To get `singles_remaining_to_take` singles,
                    # we need to break `ceil(singles_remaining_to_take / 2)` pairs.
                    # Integer division trick for ceiling: (n + d - 1) // d for n/d. Here d=2.
                    pairs_to_break = (singles_remaining_to_take + 1) // 2

                    # Consume these pairs (from the ones remaining after side-use).
                    available_pairs -= pairs_to_break

                    # We successfully formed a palindrome.
                    palindromes_made += 1
                else:
                    # Not enough total singles capacity for the center(s) *after* satisfying the pair requirement.
                    # Cannot form this palindrome. Since lengths are sorted, we likely cannot
                    # form any subsequent (longer) palindromes either as they would require
                    # at least as many pairs (or more total characters).
                    break # Exit the loop early

            else:
                # Not enough pairs from the 'available_pairs' pool for the symmetric sides.
                # Cannot form this palindrome. Since lengths are sorted, we cannot
                # form any subsequent (longer) palindromes either.
                break # Exit the loop early

        return palindromes_made