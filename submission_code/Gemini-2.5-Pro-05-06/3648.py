import math

class Solution:
  def maxCollectedFruits(self, fruits: list[list[int]]) -> int:
    n = len(fruits)
    # n is the dimension of the grid.
    # Number of moves is n-1. Max grid index is n-1.
    N = n - 1 

    if N == 0: # Grid is 1x1, 0 moves
        return fruits[0][0]

    # dp_prev stores max fruits collected up to step k-1.
    # dp_prev[c2_col][r3_row] means:
    # C1 was at (k-1, k-1)
    # C2 was at (k-1, c2_col)
    # C3 was at (r3_row, k-1)
    # Initialize for k=0:
    # C1 at (0,0), C2 at (0,N), C3 at (N,0).
    # dp_prev indices for k=0 state: c2_col_at_k0 = N, r3_row_at_k0 = N.
    dp_prev = [[-1] * (N + 1) for _ in range(N + 1)]
    
    # Fruits collected by C1, C2, C3 at their starting positions (step 0)
    initial_fruits = fruits[0][0] + fruits[0][N] + fruits[N][0]
    dp_prev[N][N] = initial_fruits

    # Iterate for k from 1 to N (N moves in total)
    # k represents the current step number (children have made k moves)
    for k in range(1, N + 1):
      dp_curr = [[-1] * (N + 1) for _ in range(N + 1)]
      
      # At current step k:
      # Child 1 is at (k,k).
      # Child 2 is at (k, c2_curr_col).
      # Child 3 is at (r3_curr_row, k).

      # Valid range for c2_curr_col at step k: [min_coord_val_curr_k, N]
      # Valid range for r3_curr_row at step k: [min_coord_val_curr_k, N]
      # min_coord_val_curr_k = max(k, N-k) due to start and end constraints.
      min_coord_val_curr_k = max(k, N - k)

      for c2_curr_col in range(min_coord_val_curr_k, N + 1):
        for r3_curr_row in range(min_coord_val_curr_k, N + 1):
          
          # Calculate fruits collected by the three children at their current cells for this step k
          fruits_at_this_step_k = 0
          current_positions = set() # Use a set to handle cases where children are in the same room
          current_positions.add((k, k))            # C1 position
          current_positions.add((k, c2_curr_col))  # C2 position
          current_positions.add((r3_curr_row, k))  # C3 position
          
          for r_pos, c_pos in current_positions:
            fruits_at_this_step_k += fruits[r_pos][c_pos]

          max_prev_dp_sum = -1 # Max sum from any valid previous state

          # Consider previous positions at step k-1:
          # C1 was at (k-1, k-1)
          # C2 was at (k-1, c2_prev_col)
          # C3 was at (r3_prev_row, k-1)
          
          # Valid range for c2_prev_col at step k-1: [min_coord_val_prev_k, N]
          # Valid range for r3_prev_row at step k-1: [min_coord_val_prev_k, N]
          min_coord_val_prev_k: int
          if k - 1 == 0: # At step 0, C2 must be at col N, C3 must be at row N
              min_coord_val_prev_k = N 
          else: # k-1 > 0
              min_coord_val_prev_k = max(k - 1, N - (k - 1))

          # Iterate over possible previous positions (c2_prev_col, r3_prev_row) for C2 and C3
          # C2 moved from (k-1, c2_prev_col) to (k, c2_curr_col)
          # C3 moved from (r3_prev_row, k-1) to (r3_curr_row, k)
          # So, c2_prev_col must be c2_curr_col + delta_c2 where delta_c2 is -1, 0, or 1.
          # Similarly for r3_prev_row.
          for delta_c2_from_prev in [-1, 0, 1]: 
            c2_prev_col = c2_curr_col + delta_c2_from_prev
            for delta_r3_from_prev in [-1, 0, 1]: 
              r3_prev_row = r3_curr_row + delta_r3_from_prev

              # Check if (c2_prev_col, r3_prev_row) is a valid state from step k-1:
              # 1. Column/Row indices must be within absolute grid bounds [0, N]
              if not (0 <= c2_prev_col <= N and 0 <= r3_prev_row <= N):
                continue
              
              # 2. Must be in the allowed 'corridor' for step k-1
              #    This means c2_prev_col >= min_coord_val_prev_k (and <=N, covered by dp_prev table size)
              #    and r3_prev_row >= min_coord_val_prev_k (and <=N)
              if not (c2_prev_col >= min_coord_val_prev_k and \
                      r3_prev_row >= min_coord_val_prev_k):
                  continue
              
              if dp_prev[c2_prev_col][r3_prev_row] != -1: # If previous state was reachable
                max_prev_dp_sum = max(max_prev_dp_sum, dp_prev[c2_prev_col][r3_prev_row])
          
          if max_prev_dp_sum != -1: # If current state is reachable from any valid previous state
            dp_curr[c2_curr_col][r3_curr_row] = max_prev_dp_sum + fruits_at_this_step_k
            
      dp_prev = dp_curr # Move current DP results to be previous for next iteration

    # Final answer is for state at step N. All children must be at (N,N).
    # C1: (N,N)
    # C2: (N, c2_col=N)
    # C3: (r3_row=N, N)
    # After the loop, dp_prev holds results for step N.
    # The required state is (c2_col=N, r3_row=N).
    # If dp_prev[N][N] is -1, it means (N,N) is unreachable (e.g. if all fruits are 0 and initial sum is 0,
    # but the problem constraints say fruits >= 0, so sum will be >=0).
    # If all fruits are 0, the result must be 0.
    return dp_prev[N][N] if dp_prev[N][N] != -1 else 0