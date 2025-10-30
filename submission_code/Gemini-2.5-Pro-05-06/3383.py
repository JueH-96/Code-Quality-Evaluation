from typing import List

class Solution:
  def maximumEnergy(self, energy: List[int], k: int) -> int:
    n = len(energy)
    
    # dp[i] will store the total energy gained if starting at magician i
    # and following the sequence i, i+k, i+2k, ...
    # Initialize with zeros; values will be filled in.
    dp = [0] * n 
    
    # Iterate from the end of the array backwards.
    # This order ensures that when we calculate dp[i], the value dp[i+k] 
    # (if i+k is a valid index) has already been computed.
    for i in range(n - 1, -1, -1):
      # Energy from the current magician i.
      current_magician_val = energy[i]
      
      # Index of the next magician in the sequence.
      next_magician_idx = i + k
      
      if next_magician_idx < n:
        # If there is a next magician in this path (i.e., i+k is a valid index),
        # the total energy starting from i is energy[i] plus the total energy 
        # starting from (i+k), which is stored in dp[next_magician_idx].
        dp[i] = current_magician_val + dp[next_magician_idx]
      else:
        # If i+k is out of bounds, magician i is the last one in this path.
        # So, the total energy is just energy[i].
        dp[i] = current_magician_val
        
    # The problem asks for the maximum possible energy. We can start at any magician.
    # Each dp[i] stores the total energy for a path starting at magician i.
    # So, the answer is the maximum value in the dp array.
    # Constraints: 1 <= energy.length, so n >= 1. Thus, dp will not be empty.
    return max(dp)