from typing import List

class Solution:
  def minChanges(self, nums: List[int], k: int) -> int:
    n = len(nums)
    m = n // 2 # Number of pairs

    # freq[d] stores the number of pairs (nums[i], nums[n-1-i])
    # such that abs(nums[i] - nums[n-1-i]) == d.
    # Size k+1 because difference can range from 0 to k.
    freq = [0] * (k + 1)

    # M_val_counts[val] stores the number of pairs (nums[i], nums[n-1-i])
    # such that M_pair = val.
    # M_pair = max(nums[i], k - nums[i], nums[n-1-i], k - nums[n-1-i]).
    # This M_pair is the maximum absolute difference achievable for this pair
    # if we are allowed to change one element of the pair to any value in [0, k]
    # while keeping the other element fixed.
    # Size k+1 because M_pair can range from 0 to k.
    M_val_counts = [0] * (k + 1)

    for i in range(m):
      u = nums[i]
      v = nums[n - 1 - i]

      # Calculate current difference for the pair
      diff = abs(u - v)
      freq[diff] += 1

      # Calculate M_pair for this pair
      # M_pair is max(max_abs_diff_if_u_fixed, max_abs_diff_if_v_fixed)
      # max_abs_diff_if_u_fixed = max(abs(u-0), abs(u-k)) = max(u, k-u)
      # max_abs_diff_if_v_fixed = max(abs(v-0), abs(v-k)) = max(v, k-v)
      # So, M_pair = max(max(u, k-u), max(v, k-v)), which is max(u, k-u, v, k-v)
      M_pair = max(u, k - u, v, k - v)
      M_val_counts[M_pair] += 1

    # count_M_less_than_X[X_val_target] will store the number of pairs (u,v)
    # such that M_pair < X_val_target.
    # This is sum_{val=0}^{X_val_target-1} M_val_counts[val].
    # Size k+1.
    count_M_less_than_X = [0] * (k + 1)
    
    # Calculate prefix sums for M_val_counts
    # current_sum_of_M_val_counts will store sum_{val=0}^{loop_idx-1} M_val_counts[val]
    # when assigning to count_M_less_than_X[loop_idx]
    current_sum_of_M_val_counts = 0
    for loop_idx in range(k + 1): # loop_idx from 0 to k
      count_M_less_than_X[loop_idx] = current_sum_of_M_val_counts
      current_sum_of_M_val_counts += M_val_counts[loop_idx]
    
    min_total_changes = m * 2 # Initialize with a safe upper bound (max possible changes, i.e., 2 per pair)

    # Iterate over all possible values for the target difference X_target
    for X_target in range(k + 1): # X_target from 0 to k
      # Cost for a given X_target:
      # For each pair (u,v):
      #   - if abs(u-v) == X_target: 0 changes.
      #   - if abs(u-v) != X_target:
      #     - if X_target <= M_pair: 1 change. (M_pair is specific to u,v)
      #     - if X_target > M_pair: 2 changes.
      #
      # Total cost can be rewritten as:
      # Sum_{pairs where abs(u-v) != X_target} (1) + 
      # Sum_{pairs where abs(u-v) != X_target AND X_target > M_pair} (1)
      #
      # The first sum term is (m - freq[X_target]). These are pairs needing at least 1 change.
      #
      # The second sum term counts pairs needing an additional change (i.e., total 2 changes).
      # These are pairs where abs(u-v) != X_target AND X_target > M_pair.
      # An important property: if abs(u-v) = X_target, then M_pair >= X_target.
      # This implies that if X_target > M_pair, then abs(u-v) cannot be X_target.
      # So, (abs(u-v) != X_target AND X_target > M_pair) is equivalent to (X_target > M_pair).
      # The number of pairs where X_target > M_pair is precisely count_M_less_than_X[X_target].
      
      cost_for_X_target = (m - freq[X_target]) + count_M_less_than_X[X_target]
      min_total_changes = min(min_total_changes, cost_for_X_target)
      
    return min_total_changes