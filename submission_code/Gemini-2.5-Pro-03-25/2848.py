from typing import List

# Define the modulo constant
MOD = 10**9 + 7

class Solution:
  """
  Solves the Special Permutations problem using Dynamic Programming with Memoization.
  Finds the number of permutations of nums such that adjacent elements satisfy the divisibility condition.
  """
  def specialPerm(self, nums: List[int]) -> int:
      """
      Calculates the total number of special permutations of the given array nums.
      A permutation is special if for all adjacent elements nums[i] and nums[i+1],
      either nums[i] % nums[i+1] == 0 or nums[i+1] % nums[i] == 0.
      Args:
          nums: A list of distinct positive integers. The length n satisfies 2 <= n <= 14.
                 Each element nums[i] satisfies 1 <= nums[i] <= 10^9.
      Returns:
          The total number of special permutations modulo 10^9 + 7.
      """
      n = len(nums)
      
      # Precompute divisibility relationships between pairs of numbers.
      # adj[i][j] is True if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0.
      # This precomputation avoids repeated modulo operations inside the recursive calls,
      # potentially speeding up the process.
      adj = [[False] * n for _ in range(n)]
      for i in range(n):
          # We only need to check pairs (i, j) where i < j due to symmetry.
          for j in range(i + 1, n): 
              if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                  adj[i][j] = True
                  adj[j][i] = True # The divisibility relationship is symmetric.

      # Memoization table (DP table) using a 2D list.
      # dp[mask][last_idx] stores the number of special permutations using the subset of elements
      # represented by the bitmask 'mask', and ending with the element nums[last_idx].
      # Initialize with -1 to signify that the state has not been computed yet.
      # The size is (2^n) x n.
      dp = [[-1] * n for _ in range(1 << n)]

      def solve(mask, last_idx):
          """
          Recursive helper function with memoization (Top-Down DP) to compute the number of special permutations.
          Args:
              mask: An integer bitmask where the k-th bit is set if nums[k] is included in the current partial permutation.
              last_idx: The index (in the original nums array) of the last element added to the current partial permutation.
          Returns:
              The number of special permutations corresponding to the state defined by (mask, last_idx), modulo MOD.
          """
          # Check if the result for this state (mask, last_idx) is already computed and stored in the DP table.
          # If dp[mask][last_idx] is not -1, it means we have computed this state before, so return the stored value.
          if dp[mask][last_idx] != -1:
              return dp[mask][last_idx]

          # Base case: If the mask contains only one set bit (specifically, the bit at last_idx),
          # it represents a permutation of length 1 consisting only of nums[last_idx].
          # There is exactly one way to form such a permutation: [nums[last_idx]].
          # The condition `mask == (1 << last_idx)` checks if only the `last_idx`-th bit is set in the mask.
          if mask == (1 << last_idx):
              return 1

          # Initialize count for the number of ways to reach the current state (mask, last_idx).
          count = 0
          
          # Calculate the mask representing the state before adding the element nums[last_idx].
          # This is the set of elements used in the permutation just before appending nums[last_idx].
          # It's obtained by unsetting the bit corresponding to last_idx in the current mask using XOR.
          prev_mask = mask ^ (1 << last_idx)
          
          # Iterate through all possible indices `prev_idx` (from 0 to n-1).
          # `prev_idx` represents the index of the element that could have been immediately preceding `nums[last_idx]`.
          for prev_idx in range(n):
              # We need to check two conditions to consider `nums[prev_idx]` as the predecessor:
              # 1. `nums[prev_idx]` must be part of the permutation represented by `prev_mask`.
              #    This is checked by testing if the `prev_idx`-th bit is set in `prev_mask`. `(prev_mask >> prev_idx) & 1` extracts the bit.
              # 2. `nums[prev_idx]` and `nums[last_idx]` must satisfy the special condition (one divides the other).
              #    This is efficiently checked using the precomputed `adj` matrix: `adj[prev_idx][last_idx]`.
              
              if (prev_mask >> prev_idx) & 1 and adj[prev_idx][last_idx]:
                  # If both conditions are met, it means we can form the current permutation ending at `nums[last_idx]`
                  # by extending a valid permutation ending at `nums[prev_idx]` (using elements in `prev_mask`).
                  # We recursively call `solve` for the subproblem state (prev_mask, prev_idx) to find the number of ways
                  # to form such prefix permutations.
                  # Add the result of the recursive call to the current count. Remember to apply modulo arithmetic.
                  count = (count + solve(prev_mask, prev_idx)) % MOD
          
          # Memoize the computed result for the current state (mask, last_idx) in the DP table before returning.
          # This ensures that we don't recompute the same state multiple times.
          dp[mask][last_idx] = count
          return count

      # Main part of the function execution starts here.
      # Calculate the final total count of special permutations of length `n`.
      # This is done by summing up the counts of permutations ending with each possible element `nums[last_idx]`,
      # where the permutation uses all elements from the original `nums` array (represented by `full_mask`).
      total_count = 0
      
      # `full_mask` is a bitmask with all n bits set to 1 (e.g., for n=3, it's 111 in binary, which is 7).
      # It represents the set containing all elements from `nums`.
      full_mask = (1 << n) - 1 
      
      # Iterate through all possible indices `last_idx` (from 0 to n-1).
      # Each `last_idx` corresponds to a potential last element `nums[last_idx]` of a full special permutation.
      for last_idx in range(n):
          # Call the `solve` function for the state (full_mask, last_idx) to get the count of special permutations
          # of length n ending with `nums[last_idx]`.
          # Add this count to the `total_count`. Apply modulo arithmetic at each addition.
          total_count = (total_count + solve(full_mask, last_idx)) % MOD
            
      # Return the final total count, which represents the total number of special permutations of `nums`.
      return total_count