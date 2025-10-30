import collections # Not strictly necessary, but useful for defaultdict if needed
from typing import List

class Solution:
  """
  Solves the problem of finding the minimum changes to satisfy the array condition using an efficient O(N+K) approach.
  The condition requires that for some integer X, abs(nums[i] - nums[n - 1 - i]) = X for all pairs (i, n-1-i).
  We can change any element nums[i] to any value in [0, k].
  """
  def minChanges(self, nums: List[int], k: int) -> int:
      """
      Calculates the minimum number of changes required.

      Args:
          nums: The input integer array of even size n.
          k: The maximum value allowed for elements after changes.

      Returns:
          The minimum number of changes required.
      """
      
      n = len(nums)
      m = n // 2  # Number of pairs (nums[i], nums[n - 1 - i])

      # count_arr[x] stores the number of pairs (a, b) such that abs(a - b) == x
      # The difference x can range from 0 to k.
      # Example: if pair is (1, 3) and k=4, diff = 2. count_arr[2] increases by 1.
      count_arr = [0] * (k + 1)
      
      # freq_M[val] stores the number of pairs i for which M_i == val.
      # M_i = max(max(a_i, b_i), k - min(a_i, b_i)) represents the maximum absolute difference 
      # achievable for pair i by changing at most one of its elements to a value in [0, k].
      # Example: if pair is (1, 3) and k=4, M = max(max(1,3), 4 - min(1,3)) = max(3, 4-1) = max(3, 3) = 3. freq_M[3] increases by 1.
      freq_M = [0] * (k + 1)

      # Process each pair (nums[i], nums[n - 1 - i])
      for i in range(m):
          a = nums[i]
          b = nums[n - 1 - i]
          
          # Calculate the initial difference for the pair
          diff = abs(a - b)
          # Since a, b are in [0, k], diff must be in [0, k].
          count_arr[diff] += 1
          
          # Calculate M_i for the pair
          # M_i is the max possible difference if we change at most one element.
          M_i = max(max(a, b), k - min(a, b))
          # M_i is guaranteed to be in [0, k].
          freq_M[M_i] += 1

      # G[X] will store the number of pairs i such that X <= M_i.
      # This counts the number of pairs for which a target difference X is achievable with at most 1 change.
      # We compute G[X] for all X using suffix sums on the freq_M array.
      # G[X] = sum(freq_M[val] for val from X to k)
      G = [0] * (k + 1)
      
      # Compute suffix sums efficiently. Start from k and go down to 0.
      if k >= 0: # Ensure k is non-negative (constraint k >= 0 holds)
           # The number of pairs with M_i >= k is simply freq_M[k].
           G[k] = freq_M[k] 
           # Iterate downwards to compute the rest using the relation G[X] = G[X+1] + freq_M[X]
           for X in range(k - 1, -1, -1):
               G[X] = G[X + 1] + freq_M[X]
      
      # Determine the minimum changes required:
      # Consider the cost for a fixed target difference X.
      # For a pair (a_i, b_i):
      # Cost is 0 if abs(a_i - b_i) == X (no changes needed).
      # Cost is 1 if abs(a_i - b_i) != X but X <= M_i (one change is sufficient).
      # Cost is 2 if abs(a_i - b_i) != X and X > M_i (two changes are needed).
      #
      # We can rephrase this using "savings" from a baseline cost of 2 per pair.
      # Total baseline cost = 2 * m.
      # For a target difference X, the total savings = sum over all pairs i of:
      #   (1 if X <= M_i) + (1 if X == abs(a_i - b_i))
      # Total savings for X = |{i | X <= M_i}| + |{i | X == abs(a_i - b_i)}|
      # Total savings for X = G[X] + count_arr[X]
      # Let F(X) = G[X] + count_arr[X] be the total savings for target difference X.
      # We want to find the X that maximizes F(X).
      # The minimum total cost = 2*m - max(F(X)) for X in [0, k].
      
      max_F = 0 # Stores the maximum savings found across all possible X
      for X in range(k + 1):
          # Calculate total savings F(X) if we choose X as the target difference
          F_X = G[X] + count_arr[X] 
          # Update the maximum savings found
          if F_X > max_F:
              max_F = F_X
              
      # The minimum number of changes = (Total baseline changes) - (Maximum possible savings)
      min_changes = m * 2 - max_F
      
      return min_changes