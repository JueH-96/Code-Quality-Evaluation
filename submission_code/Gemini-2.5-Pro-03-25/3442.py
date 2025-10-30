import math # math is not technically needed, but was imported during thought process. Can be removed.
from typing import List

class Solution:
  """
  This class provides a solution to find the maximum total reward 
  collectible based on the given rules, using a dynamic programming approach
  with bitsets represented by large integers.
  """
  def maxTotalReward(self, rewardValues: List[int]) -> int:
    """
    Calculates the maximum total reward achievable by selecting rewards 
    following the rule: chosen reward must be greater than the current total reward.

    Args:
      rewardValues: A list of integers representing the values of rewards. 
                    Constraints: 1 <= len(rewardValues) <= 2000, 1 <= rewardValues[i] <= 2000.

    Returns:
      An integer denoting the maximum total reward collectible.
    
    Problem Breakdown:
    We start with a total reward `x = 0`. We can choose any unmarked reward `r` from `rewardValues`.
    If `r > x`, we add `r` to `x` (i.e., `x = x + r`) and mark the index corresponding to `r`.
    We repeat this process to maximize the final value of `x`.

    Approach: Dynamic Programming with Bitset
    Let `dp[k]` be a boolean indicating whether a total reward of `k` is achievable.
    We can represent this DP state using a bitset (implemented as a large integer in Python),
    where the k-th bit is 1 if sum `k` is achievable, and 0 otherwise.

    The maximum possible reward `x` is bounded. If the last reward added was `r_last`,
    the sum before adding it (`x_prev = x - r_last`) must satisfy `r_last > x_prev`.
    This implies `r_last > x - r_last`, so `2 * r_last > x`.
    Since `r_last <= max(rewardValues)`, let `max_r = max(rewardValues)`. Then `x < 2 * max_r`.
    Given `max_r <= 2000`, the maximum achievable reward `x` is less than 4000.
    Thus, the bitset will need approximately 4000 bits, which Python's large integers can handle.

    Algorithm:
    1. Preprocessing: Sort `rewardValues` and remove duplicates. This simplifies the process
       as we only need to consider each unique reward value once. Let the sorted unique
       values be `unique_rewards`.
    2. Initialization: Initialize the DP state `dp = 1`. This represents that initially,
       only a sum of 0 is achievable (the 0-th bit is set).
    3. Iteration: Iterate through each unique reward value `v` in `unique_rewards`.
       For each `v`, we update the set of reachable sums. A new sum `x + v` becomes reachable
       if `x` was already reachable and `v > x`.
       This update can be efficiently performed using bitwise operations:
       - Identify reachable sums `x < v`: This corresponds to the bits in `dp` from index 0 to `v-1`.
         We can get these bits using `dp & ((1 << v) - 1)`.
       - Calculate new sums `x + v`: Left-shift the identified bits by `v`. `(dp & ((1 << v) - 1)) << v`.
       - Update `dp`: Add these new sums to the set of reachable sums using bitwise OR.
         `dp |= (dp & ((1 << v) - 1)) << v`.
    4. Result: After iterating through all unique rewards, the final `dp` bitset represents all
       achievable sums. The maximum achievable sum is the index of the highest set bit.
       This can be found using `dp.bit_length() - 1`.
    """

    # Step 1: Sort and get unique reward values.
    unique_rewards = sorted(list(set(rewardValues)))
    
    # Step 2: Initialize DP state using a bitset (large integer).
    # dp = 1 represents {0}, meaning sum 0 is initially reachable.
    dp = 1 
    
    # Step 3: Iterate through unique rewards and update DP state.
    for v in unique_rewards:
        # Calculate the part of dp representing sums x < v.
        # mask = (1 << v) - 1 isolates bits from 0 to v-1.
        mask = (1 << v) - 1
        filtered_dp = dp & mask
        
        # Calculate the new sums (x + v) by shifting.
        new_sums = filtered_dp << v
        
        # Update dp by adding the new reachable sums.
        dp |= new_sums

        # Note: The combined operation is dp |= (dp & ((1 << v) - 1)) << v

    # Step 4: Determine the maximum reachable reward.
    # The maximum reward is the index of the highest set bit in dp.
    # int.bit_length() gives the number of bits required, which is 1 + highest set bit index.
    return dp.bit_length() - 1