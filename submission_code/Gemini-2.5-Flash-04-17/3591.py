from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        alphabet_size = 26

        # Precompute costs for shifting k steps starting from each character index i
        # _cost_from[i][k]: cost of k shifts starting from alphabet index i
        # k ranges from 0 to 25. Index k means k steps.
        # We need up to 25 steps in either direction to reach any other character.
        next_cost_from = [[0] * alphabet_size for _ in range(alphabet_size)]
        prev_cost_from = [[0] * alphabet_size for _ in range(alphabet_size)]

        for start_char_index in range(alphabet_size):
            # Precompute next costs for k steps (k from 1 to 25)
            current_alphabet_index = start_char_index
            for k in range(1, alphabet_size): # k steps from 1 to 25
                # The cost of the k-th step (after k-1 steps) is based on the character
                # that is reached after k-1 steps, which is current_alphabet_index.
                next_cost_from[start_char_index][k] = next_cost_from[start_char_index][k-1] + nextCost[current_alphabet_index]
                # After shifting FROM `current_alphabet_index`, the character becomes `(current_alphabet_index + 1) % alphabet_size`.
                # This new index is the source for the *next* shift in the sequence.
                current_alphabet_index = (current_alphabet_index + 1) % alphabet_size

            # Precompute previous costs for k steps (k from 1 to 25)
            current_alphabet_index = start_char_index
            for k in range(1, alphabet_size): # k steps from 1 to 25
                 prev_cost_from[start_char_index][k] = prev_cost_from[start_char_index][k-1] + previousCost[current_alphabet_index]
                 # After shifting FROM `current_alphabet_index`, the character becomes `(current_alphabet_index - 1 + alphabet_size) % alphabet_size`.
                 # This new index is the source for the *next* shift in the sequence.
                 current_alphabet_index = (current_alphabet_index - 1 + alphabet_size) % alphabet_size


        total_cost = 0
        n = len(s)

        for i in range(n):
            start_index = ord(s[i]) - ord('a')
            target_index = ord(t[i]) - ord('a')

            # Calculate the difference in positions on the cyclic alphabet
            # diff represents the number of 'next' shifts to get from start_index to target_index
            # Range of diff is 0 to 25
            diff = (target_index - start_index + alphabet_size) % alphabet_size

            if diff == 0:
                # s[i] is already t[i], cost is 0 for this character
                cost_for_char = 0
            else:
                # Option A: Transform using 'diff' number of 'next' shifts
                # diff is between 1 and 25
                cost_next_shifts = next_cost_from[start_index][diff]

                # Option B: Transform using '26 - diff' number of 'previous' shifts
                num_prev_shifts = alphabet_size - diff
                # num_prev_shifts will be between 1 and 25 if diff is between 1 and 25
                cost_prev_shifts = prev_cost_from[start_index][num_prev_shifts]

                # The minimum cost for this character is the minimum of the two options
                cost_for_char = min(cost_next_shifts, cost_prev_shifts)

            total_cost += cost_for_char

        return total_cost