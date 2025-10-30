import collections
import math

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)

        # Handle the base case where k = 1
        # If there's at least one '1', Alice can pick it up for 0 moves by standing there.
        # If all elements are '0's, Alice must create a '1'.
        # To get the first '1' at aliceIndex (e.g., 0): create a '1' at an adjacent index (e.g., 1), cost 1 move + 1 maxChanges.
        # Then swap it to aliceIndex, cost 1 move. Total 2 moves.
        if k == 1:
            if 1 in nums:
                return 0
            else:
                return 2

        # Precompute prefix sums for indices and counts of zeros.
        # pos_sum[i] stores sum of indices from 0 to i-1.
        # zero_count[i] stores count of zeros from 0 to i-1.
        pos_sum = [0] * (n + 1)
        zero_count = [0] * (n + 1)

        for i in range(n):
            pos_sum[i+1] = pos_sum[i] + i
            zero_count[i+1] = zero_count[i]
            if nums[i] == 0:
                zero_count[i+1] += 1

        min_total_moves = float('inf')

        # Sliding window approach:
        # We consider all possible contiguous windows of `k` elements in `nums`.
        # For each window [L, R] (inclusive), Alice chooses `alice_index` as the median of the window's indices.
        for R in range(k - 1, n):
            L = R - k + 1
            alice_index = L + (k - 1) // 2

            current_moves = 0
            
            # Calculate sum of distances for elements in [L, R] relative to alice_index.
            # Elements to the left of alice_index: [L, alice_index - 1]
            sum_indices_left = pos_sum[alice_index] - pos_sum[L]
            count_indices_left = alice_index - L
            current_moves += (alice_index * count_indices_left - sum_indices_left)

            # Elements to the right of alice_index: [alice_index + 1, R]
            sum_indices_right = pos_sum[R+1] - pos_sum[alice_index+1]
            count_indices_right = R - alice_index
            current_moves += (sum_indices_right - alice_index * count_indices_right)

            # Calculate the number of '0's within the current window [L, R].
            # Each '0' in the window (excluding alice_index if it's a '1') needs to be converted.
            # This conversion costs 1 move (Type 1 action) and 1 maxChanges budget.
            zeros_in_window = zero_count[R+1] - zero_count[L]
            
            # This variable tracks the total maxChanges budget required for this window.
            current_changes_needed = zeros_in_window
            
            # Add 1 move for each '0' that needs to be converted.
            # This covers the '1 + |x - aliceIndex|' part for x != aliceIndex, nums[x]==0
            # And also the '1' for nums[alice_index]==0 before adjustment to 2.
            current_moves += zeros_in_window 

            # Adjust moves for the 'alice_index' itself based on its initial value.
            if nums[alice_index] == 0:
                # If nums[alice_index] was initially '0', it costs 2 moves to pick up (1 to create nearby, 1 to swap).
                # The '1' move from `current_moves += zeros_in_window` already accounts for part of it.
                # We need to add an extra 1 move to make it 2.
                current_moves += 1
            # If nums[alice_index] was initially '1', it costs 0 moves.
            # No adjustment needed here as it wasn't counted in `zeros_in_window`.

            # If the number of changes needed for this window is within budget, update min_total_moves.
            if current_changes_needed <= maxChanges:
                min_total_moves = min(min_total_moves, current_moves)

        return min_total_moves