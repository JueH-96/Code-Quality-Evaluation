import sys 
# Setting a higher recursion depth is usually not necessary for iterative DP
# sys.setrecursionlimit(3000) 

MOD = 10**9 + 7

class Solution:
  """
  This class implements a solution to find the number of stable binary arrays
  using dynamic programming with prefix sums optimization.

  A stable binary array `arr` satisfies:
  1. Contains exactly `zero` occurrences of 0.
  2. Contains exactly `one` occurrences of 1.
  3. Each subarray of `arr` with size greater than `limit` must contain both 0 and 1.
     This is equivalent to saying there cannot be more than `limit` consecutive 0s or 1s.

  The DP approach uses two tables:
  - f[i][j]: number of stable arrays with i zeros and j ones, ending with 0.
  - g[i][j]: number of stable arrays with i zeros and j ones, ending with 1.

  Prefix sum tables `VertSumG` and `HorzSumF` are used to speed up calculations.
  """
  def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
    
    # dp table f[i][j] stores count of arrays with i zeros, j ones ending with 0
    f = [[0] * (one + 1) for _ in range(zero + 1)]
    # dp table g[i][j] stores count of arrays with i zeros, j ones ending with 1
    g = [[0] * (one + 1) for _ in range(zero + 1)]
    
    # Prefix sum tables to optimize calculation of sums
    # VertSumG[i][j] = sum(g[k][j] for k from 0 to i) % MOD
    VertSumG = [[0] * (one + 1) for _ in range(zero + 1)]
    # HorzSumF[i][j] = sum(f[i][k] for k from 0 to j) % MOD
    HorzSumF = [[0] * (one + 1) for _ in range(zero + 1)]

    # Iterate over all possible counts of zeros (i) and ones (j)
    for i in range(zero + 1):
      for j in range(one + 1):
        # Base case: empty array (i=0, j=0). Counts are 0. Handled by initialization.
        # Skip calculation for (0,0) state.
        if i == 0 and j == 0:
          continue 

        # Calculate base values: contribution from arrays made of a single block of 0s or 1s.
        # An array consisting only of `i` zeros is valid if `1 <= i <= limit`. Such an array ends in 0.
        BASE_f = 1 if j == 0 and 1 <= i <= limit else 0
        # An array consisting only of `j` ones is valid if `1 <= j <= limit`. Such an array ends in 1.
        BASE_g = 1 if i == 0 and 1 <= j <= limit else 0

        # Calculate f[i][j]: number of stable arrays with i zeros, j ones ending in 0.
        # An array ending in 0 must have been formed by appending a block of 1 to `limit` zeros 
        # to an array ending in 1.
        # Specifically, it ends with ...1 followed by k zeros, where 1 <= k <= limit.
        # The prefix ...1 has i-k zeros and j ones, ending in 1.
        # The count f[i][j] is the sum of g[i-k][j] for k from 1 to min(i, limit).
        # This sum is equivalent to sum(g[p][j] for p from max(0, i-limit) to i-1).
        if i > 0:
          # Calculate the sum using prefix sums: VertSumG[i-1][j] - VertSumG[i-limit-1][j]
          # We need sum_{p = i-min(i, limit)}^{i-1} g[p][j]
          # The index range is max(0, i-limit) to i-1
          sum_g_term = VertSumG[i-1][j] # This is sum g[p][j] for p=0..i-1
          # If i-limit > 0, we need to subtract sum g[p][j] for p=0..i-limit-1
          if i - limit - 1 >= 0:
             # Subtract the sum of terms outside the desired range (p < i-limit)
             sum_g_term = (sum_g_term - VertSumG[i-limit-1][j] + MOD) % MOD
             
          # Add the base case contribution and the sum term
          f[i][j] = (BASE_f + sum_g_term) % MOD
        else: # i == 0
          # If i=0, an array cannot end in 0 unless it's the base case of all zeros (handled by BASE_f).
          # Since j > 0 in this case (as i=0, j=0 is skipped), BASE_f is 0.
          # The sum term would involve g[-k][j], which are 0. So f[0][j] = 0.
          # The only possibility for BASE_f=1 is when j=0, which is handled correctly.
          f[i][j] = BASE_f

        # Calculate g[i][j]: number of stable arrays with i zeros, j ones ending in 1.
        # An array ending in 1 must have been formed by appending a block of 1 to `limit` ones
        # to an array ending in 0.
        # Specifically, it ends with ...0 followed by k ones, where 1 <= k <= limit.
        # The prefix ...0 has i zeros and j-k ones, ending in 0.
        # The count g[i][j] is the sum of f[i][j-k] for k from 1 to min(j, limit).
        # This sum is equivalent to sum(f[i][q] for q from max(0, j-limit) to j-1).
        if j > 0:
          # Calculate the sum using prefix sums: HorzSumF[i][j-1] - HorzSumF[i][j-limit-1]
          # We need sum_{q = j-min(j, limit)}^{j-1} f[i][q]
          # The index range is max(0, j-limit) to j-1
          sum_f_term = HorzSumF[i][j-1] # This is sum f[i][q] for q=0..j-1
          # If j-limit > 0, we need to subtract sum f[i][q] for q=0..j-limit-1
          if j - limit - 1 >= 0:
             # Subtract the sum of terms outside the desired range (q < j-limit)
             sum_f_term = (sum_f_term - HorzSumF[i][j-limit-1] + MOD) % MOD

          # Add the base case contribution and the sum term
          g[i][j] = (BASE_g + sum_f_term) % MOD
        else: # j == 0
          # If j=0, an array cannot end in 1 unless it's the base case of all ones (handled by BASE_g).
          # Since i > 0 in this case (as i=0, j=0 is skipped), BASE_g is 0.
          # The sum term would involve f[i][-k], which are 0. So g[i][0] = 0.
          # The only possibility for BASE_g=1 is when i=0, which is handled correctly.
          g[i][j] = BASE_g

        # Update prefix sums using the computed f[i][j] and g[i][j] for the current state (i, j)
        # Vertical prefix sum for g: includes g[i][j]
        VertSumG[i][j] = ((VertSumG[i-1][j] if i > 0 else 0) + g[i][j]) % MOD
        # Horizontal prefix sum for f: includes f[i][j]
        HorzSumF[i][j] = ((HorzSumF[i][j-1] if j > 0 else 0) + f[i][j]) % MOD
            
    # The final answer is the total count of stable arrays with exactly `zero` zeros and `one` ones.
    # This is the sum of counts of arrays ending in 0 (f[zero][one]) and arrays ending in 1 (g[zero][one]).
    result = (f[zero][one] + g[zero][one]) % MOD
    return result