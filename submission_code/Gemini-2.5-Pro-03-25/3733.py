import collections
from typing import List

class Solution:
  """
  Finds the length of the longest V-shaped diagonal segment in a grid.
  A V-shaped diagonal segment starts with 1, follows the sequence 2, 0, 2, 0, ...,
  moves along a diagonal direction, and can make at most one 90-degree clockwise turn
  while maintaining the sequence continuity.
  """
  def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
      """
      Calculates the length of the longest V-shaped diagonal segment using dynamic programming.

      Args:
          grid: A 2D list of integers where each element is 0, 1, or 2.

      Returns:
          The length of the longest V-shaped diagonal segment. Returns 0 if no valid segment exists.
      """
      n = len(grid)
      m = len(grid[0])

      # DP tables store lengths of segments ending at cell (r, c) moving in direction dir_idx.
      # dp1: length of standard segment (1, 2, 0, ...)
      # dp0: length of segment starting with 0 (0, 2, 0, ...)
      # dp2: length of segment starting with 2 (2, 0, 2, ...)
      # Each table is N x M x 4 dimensions.
      dp1 = [[[0] * 4 for _ in range(m)] for _ in range(n)]
      dp0 = [[[0] * 4 for _ in range(m)] for _ in range(n)]
      dp2 = [[[0] * 4 for _ in range(m)] for _ in range(n)]

      max_len = 0 # Stores the maximum length found so far

      def seq(k):
          """
          Returns the required value at index k (0-based) in the sequence 1, 2, 0, 2, 0, ...
          Returns -1 for invalid index k < 0.
          """
          if k < 0: return -1 
          if k == 0: return 1
          # For k > 0: indices 1, 3, 5... require 2; indices 2, 4, 6... require 0.
          return 2 if k % 2 == 1 else 0

      # Define the 4 diagonal directions as (dr, dc) displacement vectors for segment motion.
      # 0: TL-BR (+1, +1) Top-Left to Bottom-Right
      # 1: TR-BL (+1, -1) Top-Right to Bottom-Left
      # 2: BR-TL (-1, -1) Bottom-Right to Top-Left
      # 3: BL-TR (-1, +1) Bottom-Left to Top-Right
      dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)] 

      # Define iteration ranges for each direction to ensure DP dependencies are met.
      # For example, Dir 0 (+1, +1) depends on the cell at (r-1, c-1), so iterate r and c increasing.
      # Dir 2 (-1, -1) depends on (r+1, c+1), so iterate r and c decreasing.
      iter_ranges = [
          (range(n), range(m)),                          # Dir 0
          (range(n), range(m - 1, -1, -1)),              # Dir 1
          (range(n - 1, -1, -1), range(m - 1, -1, -1)),  # Dir 2
          (range(n - 1, -1, -1), range(m)),              # Dir 3
      ]

      # --- Compute DP tables ---
      for dir_idx in range(4): # Iterate through each of the 4 directions
          dr, dc = dirs[dir_idx] # Get displacement for this direction
          r_range, c_range = iter_ranges[dir_idx] # Get iteration order for this direction
          
          for r in r_range: # Iterate through rows based on direction's requirement
               for c in c_range: # Iterate through columns based on direction's requirement
                  pr, pc = r - dr, c - dc # Calculate coordinates of the previous cell in the path
                  valid_prev = (0 <= pr < n and 0 <= pc < m) # Check if previous cell is within grid bounds

                  # --- Update dp1: Standard sequence 1, 2, 0, ... ---
                  if grid[r][c] == 1:
                      # A '1' always starts a new standard segment of length 1
                      dp1[r][c][dir_idx] = 1
                  elif valid_prev: # If cell is not 1, check if it can extend a segment from previous cell
                      L = dp1[pr][pc][dir_idx] # Length of standard segment ending at previous cell
                      if L > 0 and grid[r][c] == seq(L): # Check if current cell value matches sequence at index L
                          dp1[r][c][dir_idx] = L + 1 # Extend the segment
                  # If grid[r][c] is not 1 and cannot extend, dp1[r][c][dir_idx] remains 0.
                  
                  # Update overall max_len found so far with length of any valid straight segment (dp1)
                  current_dp1_len = dp1[r][c][dir_idx]
                  if current_dp1_len > 0:
                       max_len = max(max_len, current_dp1_len)

                  # --- Update dp0: Sequence starting with 0 (0, 2, 0, ...) ---
                  if grid[r][c] == 0:
                      if valid_prev:
                          # To extend a sequence ending in 0, the previous cell must end a sequence ending in 2 (type dp2)
                          L = dp2[pr][pc][dir_idx] 
                          if L > 0: # If a valid segment of type dp2 ends at previous cell, extend it
                              dp0[r][c][dir_idx] = L + 1
                          else: # Cannot extend, this '0' starts a new segment of type dp0
                              dp0[r][c][dir_idx] = 1 
                      else: # Previous cell out of bounds, this '0' starts a new segment of type dp0
                         dp0[r][c][dir_idx] = 1 
                  # If grid[r][c] is not 0, dp0[r][c][dir_idx] remains 0.
                  
                  # --- Update dp2: Sequence starting with 2 (2, 0, 2, ...) ---
                  if grid[r][c] == 2:
                      if valid_prev:
                          # To extend a sequence ending in 2, the previous cell must end a sequence ending in 0 (type dp0)
                          L = dp0[pr][pc][dir_idx] 
                          if L > 0: # If a valid segment of type dp0 ends at previous cell, extend it
                              dp2[r][c][dir_idx] = L + 1
                          else: # Cannot extend, this '2' starts a new segment of type dp2
                              dp2[r][c][dir_idx] = 1 
                      else: # Previous cell out of bounds, this '2' starts a new segment of type dp2
                          dp2[r][c][dir_idx] = 1
                  # If grid[r][c] is not 2, dp2[r][c][dir_idx] remains 0.

      # --- Combine DP values to find the longest V-shape ---
      # Iterate through all cells (r, c) as potential turning points (vertices) of the V
      for r in range(n):
          for c in range(m):
              # Consider each direction 'dir1' as the direction of the first arm ending at (r, c)
              for dir1 in range(4): 
                  L1 = dp1[r][c][dir1] # Length of the first arm (must be standard sequence type dp1)
                  if L1 == 0: continue # If no valid first arm ends here in this direction, skip

                  # Calculate the direction 'dir2' for the second arm after a 90-degree clockwise turn
                  dir2 = (dir1 + 1) % 4 
                  dr2, dc2 = dirs[dir2] # Displacement for the second arm's direction
                  nr, nc = r + dr2, c + dc2 # Coordinates of the cell immediately following the vertex (r,c) along dir2

                  # Check if this next cell (nr, nc) is within the grid bounds
                  if 0 <= nr < n and 0 <= nc < m: 
                      # Determine the value required at (nr, nc) to maintain sequence continuity
                      # The vertex (r, c) is at index L1-1. The next cell (nr, nc) is at index L1.
                      required_seq_val = seq(L1) 
                      L2 = 0 # Initialize length of the second arm (starting from nr, nc)

                      # Check if the cell (nr, nc) can potentially start the required sequence type (0,2,... or 2,0,...)
                      if required_seq_val == 0:
                         # Need sequence 0, 2, 0, ... starting from (nr, nc).
                         # Look up dp0[nr][nc][dir2], which stores the length of such a segment ending at (nr, nc).
                         L2 = dp0[nr][nc][dir2] 
                      elif required_seq_val == 2:
                         # Need sequence 2, 0, 2, ... starting from (nr, nc).
                         # Look up dp2[nr][nc][dir2], which stores the length of such a segment ending at (nr, nc).
                         L2 = dp2[nr][nc][dir2]
                      # Note: required_seq_val cannot be 1 because L1 >= 1, so seq(L1) is 2 or 0.

                      if L2 > 0:
                          # If a valid second arm exists (L2 > 0), calculate total V-shape length.
                          # Total length = length of first arm (L1) + length of second arm (L2)
                          # Update the overall maximum length found.
                          max_len = max(max_len, L1 + L2)
                          
      return max_len # Return the maximum length found (could be from straight or V-shaped segment)