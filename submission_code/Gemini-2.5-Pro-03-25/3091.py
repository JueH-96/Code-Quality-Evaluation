import collections
from typing import List

class Solution:
  """
  Calculates the count of sub-multisets of nums whose sum falls within the range [l, r].
  Uses dynamic programming with an optimization for handling element counts efficiently.
  """
  def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
    """
    Args:
      nums: A list of non-negative integers.
      l: The lower bound of the sum range (inclusive).
      r: The upper bound of the sum range (inclusive).

    Returns:
      The count of sub-multisets with sum in [l, r], modulo 10^9 + 7.
    """
    MOD = 10**9 + 7
    
    # Count frequencies of each number in the input list `nums`.
    counts = collections.Counter(nums)
    
    # Handle zeros separately. The number of zeros affects the final count multiplicatively.
    # `count0` stores the frequency of 0 in `nums`.
    count0 = counts.get(0, 0) # Use .get() for safety if 0 is not present.
    
    # Remove 0 from the frequency map if it exists. 
    # The DP part will only process non-zero elements.
    if 0 in counts:
        del counts[0]
        
    # Initialize DP table. `dp[s]` will store the number of ways to form sum `s`
    # using the non-zero elements processed so far.
    # The maximum sum we need to compute up to is `r`. The DP table size is `r+1`.
    dp = [0] * (r + 1)
    dp[0] = 1  # Base case: sum 0 is possible in 1 way (by choosing the empty multiset).
    
    # Get the list of distinct non-zero elements present in `nums`.
    # The order of processing these elements doesn't matter for the final result.
    distinct_nonzero_elements = list(counts.keys())
    
    # Iterate through each distinct non-zero element type `x`.
    for x in distinct_nonzero_elements:
        # Get the count `c` (frequency) of the current element `x`.
        c = counts[x] 
        
        # Create a copy of the current DP state `dp`. 
        # `dp_old` stores the DP state *before* processing element `x`.
        # This is crucial for the correction step below.
        dp_old = dp[:] 
        
        # Update the DP table considering element `x`. Iterate through sums `s` from `x` up to `r`.
        # We iterate upwards `s` from `x`.
        for s in range(x, r + 1):
            # This update represents adding element `x` to previously formed sub-multisets.
            # `dp[s] = (dp[s] + dp[s - x]) % MOD`
            # Since `dp[s-x]` might have already been updated in this loop for the current `x`,
            # this effectively performs an update similar to the unbounded knapsack problem
            # (allowing unlimited copies of `x` initially).
            dp[s] = (dp[s] + dp[s - x]) % MOD
            
            # Correction step: The above update assumes potentially infinite copies of `x`.
            # We need to subtract the counts of sub-multisets that would have used more than `c` copies of `x`.
            # These invalid states originate from states `s - (c+1)*x` computed *before* considering `x`
            # (which are stored in `dp_old`).
            # If we add `c+1` copies of `x` to a state `dp_old[limit_idx]`, we reach sum `s`.
            # These are exactly the states counted incorrectly by the unbounded update.
            limit_idx = s - (c + 1) * x
            if limit_idx >= 0:
                # Subtract the count `dp_old[limit_idx]` from `dp[s]`.
                # Use modulo arithmetic carefully: `(a - b + MOD) % MOD` ensures a positive result.
                dp[s] = (dp[s] - dp_old[limit_idx] + MOD) % MOD
    
    # After processing all distinct non-zero elements, `dp[s]` holds the count of
    # sub-multisets formed using *only non-zero elements* that sum up to `s`.
    
    # Calculate the total count for sums within the target range [l, r].
    total_count_in_range = 0
    # Sum `dp[s]` for `s` from `l` to `r` inclusive.
    # The constraints guarantee 0 <= l <= r, so `l` and `r` are valid indices for `dp`.
    for s in range(l, r + 1):
        total_count_in_range = (total_count_in_range + dp[s]) % MOD
            
    # Final adjustment for zeros: Each valid sub-multiset (summing s in [l, r] using non-zeros)
    # can be augmented by adding 0, 1, ..., up to `count0` zeros. 
    # This means each count `dp[s]` contributes `count0 + 1` times to the final answer.
    # We multiply the total count in the range by `(count0 + 1)`.
    # Note: If l=0, dp[0]=1 (empty set using non-zeros) is included. Multiplying by (count0+1) 
    # correctly accounts for the sub-multisets consisting solely of zeros: {}, {0}, {0,0}, ...
    final_answer = (total_count_in_range * (count0 + 1)) % MOD
    
    return final_answer