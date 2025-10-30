import math
import itertools # For permutations

class Solution:
  def maximumValueSum(self, board: list[list[int]]) -> int:
    m = len(board)
    n = len(board[0])

    # Ensure m <= n by transposing if necessary.
    # This optimizes by making the cubic term depend on the smaller dimension.
    if m > n:
      # Transpose the board
      new_board = [[0] * m for _ in range(n)]
      for r_idx in range(m):
        for c_idx in range(n):
          new_board[c_idx][r_idx] = board[r_idx][c_idx]
      board = new_board
      m, n = n, m # Update dimensions

    max_total_sum = -math.inf

    # Iterate over all combinations of 3 distinct rows
    # These are r_options[0], r_options[1], r_options[2]
    # Using itertools.combinations is cleaner than nested loops for selection
    
    # chosen_rows_indices will be tuples like (r1, r2, r3)
    for chosen_rows_indices_tuple in itertools.combinations(range(m), 3):
      # chosen_rows_indices_tuple contains 3 distinct row indices
      
      # Iterate over all 3! permutations of these chosen rows.
      # Each permutation (S0, S1, S2) assigns a chosen row to a "slot".
      # Slot 0 gets column c0, Slot 1 gets c1, Slot 2 gets c2.
      for S_rows_permuted in itertools.permutations(chosen_rows_indices_tuple):
        S0, S1, S2 = S_rows_permuted[0], S_rows_permuted[1], S_rows_permuted[2]

        # For row S0, get top 3 values (board[S0][c], c)
        list_S0_items = []
        for c_idx in range(n):
          list_S0_items.append((board[S0][c_idx], c_idx))
        # Sort by value descending. If using heapq.nlargest, it's O(N log K) = O(N) for K=3
        list_S0_items.sort(key=lambda x: x[0], reverse=True)
        top_S0_choices = list_S0_items[:3] # n >= 3, so slicing up to 3 is fine.

        list_S1_items = []
        for c_idx in range(n):
          list_S1_items.append((board[S1][c_idx], c_idx))
        list_S1_items.sort(key=lambda x: x[0], reverse=True)
        top_S1_choices = list_S1_items[:3]

        list_S2_items = []
        for c_idx in range(n):
          list_S2_items.append((board[S2][c_idx], c_idx))
        list_S2_items.sort(key=lambda x: x[0], reverse=True)
        top_S2_choices = list_S2_items[:3]
        
        current_max_for_permutation = -math.inf
        
        # Try all 3*3*3 = 27 combinations of top choices
        for val_s0, c_s0 in top_S0_choices:
          for val_s1, c_s1 in top_S1_choices:
            if c_s1 == c_s0: # Columns must be distinct
              continue
            for val_s2, c_s2 in top_S2_choices:
              if c_s2 == c_s0 or c_s2 == c_s1: # Columns must be distinct
                continue
              
              current_sum = val_s0 + val_s1 + val_s2
              if current_sum > current_max_for_permutation:
                current_max_for_permutation = current_sum
        
        # If current_max_for_permutation is still -math.inf, it means no valid combination of 3 distinct columns was found.
        # However, since n >= 3, there will always be at least one combination of 3 distinct columns.
        # (e.g. (c0,c1,c2) = (0,1,2) ) unless n < 3, which is ruled out by constraints.
        if current_max_for_permutation > -math.inf: # Check to be safe, or if board values could all be -inf
             max_total_sum = max(max_total_sum, current_max_for_permutation)

    return max_total_sum