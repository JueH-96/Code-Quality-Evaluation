import math
from collections import defaultdict
from typing import List

class Solution:
  """
  Finds the maximum element-sum of a complete subset of indices based on the problem specification
  and interpretation derived from provided examples.

  Problem interpretation:
  The problem asks for the maximum sum of elements `nums[i]` where the indices `i` belong to a "complete subset".
  Based on the examples, a subset of indices {i_1, ..., i_k} is considered "complete" if the product 
  of any pair of indices i_j * i_l is a perfect square. This condition is mathematically equivalent 
  to stating that all indices in the subset must share the same square-free part.
  The square-free part of an integer is the product of its prime factors raised to the power of 1,
  considering only prime factors that appear with an odd exponent in the integer's prime factorization. 
  For example, the square-free part of 12 (2^2 * 3) is 3, and the square-free part of 72 (2^3 * 3^2) is 2.

  Algorithm:
  1. Precompute the square-free part for each index `i` from 1 to `n` (where `n` is the length of `nums`). 
     This can be done efficiently using a sieve-like method.
  2. Initialize a dictionary or hash map to store the sum of elements for each distinct square-free part found.
  3. Iterate through the indices `i` from 1 to `n`. For each index `i`:
     a. Determine its square-free part, `S = sf(i)`.
     b. Retrieve the corresponding element value `nums[i-1]` (using 0-based indexing for the input list `nums`).
     c. Add this value to the total sum associated with the square-free part `S` in the dictionary.
  4. After processing all indices, find the maximum value among all the sums stored in the dictionary. 
     This maximum sum represents the maximum element-sum achievable from a complete subset of indices.
  """
  def maximumSum(self, nums: List[int]) -> int:
      """
      Calculates the maximum element-sum of a complete subset of indices.

      Args:
          nums: A list of integers. The problem statement implies this list represents values
                of a 1-indexed array, meaning `nums[i-1]` is the value associated with index `i`.

      Returns:
          The maximum possible element-sum achievable from any "complete" subset of indices {1, ..., n}.
      """
      n = len(nums)
      
      # Constraints state 1 <= n <= 10^4. Thus, nums is never empty.

      # Step 1: Precompute square-free parts for indices 1 to n.
      # `sf[i]` will store the square-free part of integer `i`.
      # Initialize sf array where sf[i] = i initially.
      sf = list(range(n + 1)) 
      
      # Use a sieve-like method. Iterate potential prime factors p up to sqrt(n).
      # The limit needs to check factors up to floor(sqrt(n)). Add 1 to range upper bound.
      limit = int(math.sqrt(n)) + 1 
      for p in range(2, limit):
          # Calculate p squared. We will remove factors of p^2 from multiples of p^2.
          p_squared = p * p
          # Iterate through all multiples of p^2 up to n.
          for i in range(p_squared, n + 1, p_squared):
              # Keep dividing sf[i] by p_squared as long as it's divisible.
              # This removes the square factor p*p completely.
              # Example: sf[72], p=2, p_sq=4. sf[72]=72. 72%4=0 -> sf[72]=18. 18%4!=0. Stop.
              # Later, p=3, p_sq=9. Check sf[72] which is now 18. 18%9=0 -> sf[72]=2. 2%9!=0. Stop. Final sf[72]=2.
              while sf[i] % p_squared == 0:
                  sf[i] //= p_squared
      
      # Step 2 & 3: Group indices by square-free part and sum corresponding elements.
      # Use defaultdict to easily accumulate sums for each square-free part group.
      sums_by_sq_free_part = defaultdict(int)
      
      # Iterate through 1-based indices from 1 to n.
      for i in range(1, n + 1):
          # Get the precomputed square-free part of the index i.
          square_free_part = sf[i]
          
          # Access the element value using 0-based index `i-1`.
          # Add this value to the running sum for this square-free part group.
          sums_by_sq_free_part[square_free_part] += nums[i-1]
          
      # Step 4: Find the maximum sum among all groups.
      # Initialize max_sum to 0. Since nums[i] >= 1, all sums will be non-negative.
      max_sum = 0
      # Iterate through the computed sums for each group.
      # `sums_by_sq_free_part.values()` gives an iterator over all collected sums.
      for current_sum in sums_by_sq_free_part.values():
          # Update max_sum if the current group's sum is larger.
          max_sum = max(max_sum, current_sum)
              
      # Return the overall maximum sum found.
      return max_sum