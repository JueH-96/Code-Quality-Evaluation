import math
from typing import List

class Solution:
    """
    Solves the problem of finding the maximum possible minimum value in gameScore
    after at most m moves. It uses binary search on the potential minimum score `x`.
    For a given `x`, a helper function `check(x)` determines if it's possible to achieve
    a minimum score of at least `x` in `gameScore` for all indices using at most `m` moves.
    The `check` function calculates the minimum number of moves required using a two-pass 
    dynamic programming approach on the array indices.
    """
    def maxScore(self, points: List[int], m: int) -> int:
        """
        Calculates the maximum possible minimum score using binary search.

        Args:
          points: A list of integers representing points awarded at each index upon visit.
          m: The maximum number of moves allowed.

        Returns:
          The maximum possible minimum score achievable across all indices in gameScore.
        """
        n = len(points)

        def check(x: int) -> bool:
          """
          Checks if a minimum score of x is achievable for all elements in gameScore
          with at most m moves.
          This function calculates the minimum number of moves needed (L_min) to ensure 
          gameScore[i] >= x for all i, and returns True if L_min <= m, False otherwise.
          The calculation is based on required visits derived from path constraints.

          Args:
            x: The target minimum score to check.

          Returns:
            True if minimum score x is achievable within m moves, False otherwise.
          """
          # Base case: A minimum score of 0 is always achievable, even with 0 moves.
          if x == 0:
               return True 

          # Calculate required visits k_i for each index i to achieve score x.
          # k_i = ceil(x / points[i]). The formula uses integer arithmetic for precision.
          # points[i] >= 1 is guaranteed by constraints, avoiding division by zero.
          k = []
          for i in range(n):
              k.append((x + points[i] - 1) // points[i])

          # Initialize `temp_b` array. `temp_b[i]` will store the minimum visits required for index `i`
          # after considering path constraints. We start with the base requirements `k`.
          temp_b = list(k) 
          
          # Constraint: The path must start by visiting index 0. This requires at least one visit to index 0.
          temp_b[0] = max(1, temp_b[0]) 

          # Pass 1: Left to right propagation of requirements.
          # This pass enforces constraints derived from the game's movement rules (flow equations).
          # Specifically, it ensures that the number of visits `v_i` is sufficient considering dependencies from the left (i-1).
          # The update rule `temp_b[i] = max(temp_b[i], temp_b[i-1] - k[i-1])` captures this propagation.
          # Note: This update rule is based on patterns observed in similar problems involving path constraints.
          # The term `- k[i-1]` reflects how required visits at `i` depend on visits and requirements at `i-1`.
          for i in range(1, n):
               # We use k[i-1] (base requirement) not temp_b[i-1] in the subtraction part
               # This prevents potentially inflated values from overly propagating.
               temp_b[i] = max(temp_b[i], temp_b[i-1] - k[i-1]) 
          
          # Pass 2: Right to left propagation of requirements.
          # This pass enforces constraints symmetrically, considering dependencies from the right (i+1).
          # The update rule `temp_b[i] = max(temp_b[i], temp_b[i+1] - k[i+1])` ensures
          # `temp_b[i]` is sufficient given `temp_b[i+1]`.
          for i in range(n - 2, -1, -1):
               # Similar to Pass 1, we use base requirement k[i+1] in the subtraction.
               temp_b[i] = max(temp_b[i], temp_b[i+1] - k[i+1]) 

          # Calculate the minimum total moves required (L_min).
          # It's established that the total number of moves in such a process is equal to the sum of visits `sum(v_i)`.
          # Here, `temp_b` represents the minimum required visits `v_i`.
          current_sum = sum(temp_b)
          
          # Check if the calculated minimum required moves are within the allowed budget `m`.
          return current_sum <= m

        # Binary search setup
        low = 0 # The minimum possible score is 0.
        # The maximum possible score could potentially be large (up to max(points) * m approx 10^15).
        # Set a safe upper bound for binary search.
        high = 10**15 + 2 
        ans = 0 # Stores the best achievable minimum score found so far. Initialized to 0.

        # Binary search loop to find the maximum `x` for which `check(x)` is True.
        while low <= high:
          mid = (low + high) // 2
          
          # If check(mid) is true, it means a minimum score of `mid` is achievable.
          # We store `mid` as a potential answer and try to achieve an even higher minimum score.
          if check(mid):
              ans = mid 
              low = mid + 1
          # If check(mid) is false, `mid` is too high (not achievable within `m` moves).
          # We need to search for a smaller minimum score.
          else:
              high = mid - 1
        
        # Return the maximum minimum score found that is achievable.
        return ans