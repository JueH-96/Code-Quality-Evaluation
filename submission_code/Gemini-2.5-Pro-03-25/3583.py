import bisect
from typing import List

class Solution:
  """
  Solves the problem of finding the k-th smallest GCD among all pairs in an array `nums`.
  The indices `k` are given in the `queries` array (0-based index).
  The approach involves calculating the frequency of each possible GCD value `g` among all pairs,
  then computing prefix sums of these frequencies. This allows efficient lookup for the k-th smallest GCD
  using binary search for each query.

  The core idea is based on number theory:
  1. Count frequencies of numbers in `nums`.
  2. For each `k`, compute `C[k]`, the count of numbers in `nums` divisible by `k`.
  3. Compute `count_div[k] = C[k] * (C[k] - 1) // 2`, the number of pairs `(nums[i], nums[j])` where both elements are divisible by `k`. This counts pairs whose GCD is a multiple of `k`.
  4. Use the relationship `count_div[g] = Sum_{k: g|k} exact_count[k]` to find `exact_count[g]`, the number of pairs whose GCD is exactly `g`. This is done iteratively from `M` down to 1, effectively using inclusion-exclusion. `exact_count[g] = count_div[g] - Sum_{k: g|k, k > g} exact_count[k]`.
  5. Compute prefix sums of `exact_count`. `prefix_sum[G]` stores the total count of pairs with GCD `<= G`.
  6. For each query `q`, find the smallest `G` such that `prefix_sum[G] >= q + 1`. This `G` is the `(q+1)`-th smallest GCD value (corresponding to index `q`). Binary search (`bisect_left`) on `prefix_sum` finds `G` efficiently.

  Time complexity: O(N + M log M + Q log M), where N = len(nums), M = max(nums), Q = len(queries).
  The dominant parts are step 2 (calculating `C[k]`, M log M), step 4 (calculating `exact_count`, M log M) and step 6 (processing queries, Q log M). Step 1 is O(N). Steps 3 and 5 are O(M).
  Space complexity: O(N + M), primarily for storing number frequencies (`counts_arr`, size M+1, potentially sparse but depends on N distinct values too) and intermediate counts (`C`, `count_div`, `exact_count`, `prefix_sum`, all size M+1). 
  If N is much larger than M, space could be considered O(N). If M is larger, O(M). So O(N+M) covers both.
  """
  def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
      
      # Determine the maximum value M in nums. Constraints: nums[i] >= 1, N >= 2.
      M = 0
      # Constraints guarantee N >= 2, so nums is not empty and has at least two elements.
      # max() is safe to use.
      if nums: # Added check just in case constraints are violated or tests include empty/small lists
          M = max(nums)
      
      # Handle edge case where M=0 if nums contains only 0s.
      # Constraint nums[i] >= 1 ensures M >= 1.
      if M == 0:
          # If M is 0, all numbers must be 0. GCD(0,0) is typically undefined or considered 0.
          # Problem constraints state nums[i] >= 1. This case shouldn't happen.
          # If it could happen, need clarification on GCD(0,0) and behavior.
          # Assuming constraints hold, M >= 1.
           pass 

      # 1. Count frequency of each number present in nums.
      # Array size M+1, indices represent numbers from 0 to M.
      counts_arr = [0] * (M + 1)
      for x in nums:
          # No need to check x <= M because M is max(nums)
          counts_arr[x] += 1

      # 2. Compute C[k]: count of numbers in nums divisible by k.
      # This uses a sieve-like approach over multiples.
      C = [0] * (M + 1)
      for k in range(1, M + 1):
          # Iterate through multiples m of k up to M
          for m in range(k, M + 1, k):
              # Add the frequency of number m to C_k
              C[k] += counts_arr[m] 

      # 3. Compute count_div[k]: count of pairs (i, j) where k divides gcd(nums[i], nums[j]).
      # This is equivalent to the number of pairs where both nums[i] and nums[j] are multiples of k.
      # Calculated as Combinations(C[k], 2).
      count_div = [0] * (M + 1)
      for k in range(1, M + 1):
          if C[k] >= 2:
              count_div[k] = C[k] * (C[k] - 1) // 2 # Use integer division

      # 4. Compute exact_count[g]: count of pairs (i, j) where gcd(nums[i], nums[j]) = g.
      # Uses inclusion-exclusion principle / Mobius inversion logic implicitly.
      # Calculated iteratively from M down to 1.
      exact_count = [0] * (M + 1)
      for g in range(M, 0, -1):
          # Start with count_div[g], which counts pairs whose GCD is g OR a multiple of g.
          # Subtract counts of pairs whose GCD is a multiple of g > g.
          # The values exact_count[k] for k > g and g|k have already been computed.
          sum_multiples = 0
          for k in range(2 * g, M + 1, g):
                 sum_multiples += exact_count[k]
          
          # Calculate exact_count[g]
          exact_count[g] = count_div[g] - sum_multiples

      # 5. Compute prefix sums of exact_count.
      # prefix_sum[G] = total number of pairs with GCD <= G.
      prefix_sum = [0] * (M + 1)
      for g in range(1, M + 1):
          prefix_sum[g] = prefix_sum[g-1] + exact_count[g]
      
      # 6. Process queries using the computed prefix sums.
      ans = []
      for q in queries:
          # Query q asks for the element at index q (0-based) in the sorted list of GCDs.
          # This is the (q+1)-th smallest GCD value.
          # We need to find the smallest G such that the number of pairs with GCD <= G
          # is at least q + 1. In terms of prefix sums: find smallest G s.t. prefix_sum[G] >= q + 1.
          target = q + 1
          
          # Use binary search (bisect_left) on the prefix_sum array.
          # bisect_left finds the first index G where prefix_sum[G] >= target.
          G = bisect.bisect_left(prefix_sum, target)
          
          # G is the smallest index satisfying the condition, which is the required GCD value.
          # Constraints ensure G is valid and within [1, M].
          # 0 <= q < n*(n-1)/2 implies 1 <= target <= n*(n-1)/2 = prefix_sum[M].
          # bisect_left on non-decreasing array guarantees finding a valid index.
          # Since prefix_sum[0]=0 and target >=1, G >= 1.
          # Since target <= prefix_sum[M], G <= M.
          
          ans.append(G)
            
      return ans