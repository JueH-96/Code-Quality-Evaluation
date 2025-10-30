import sys 
from typing import List

# Setting recursion depth is not needed for this iterative solution
# sys.setrecursionlimit(2000) 

MOD = 10**9 + 7

class Solution:
  """
  Calculates the number of monotonic pairs (arr1, arr2) based on the given nums array.
  A pair (arr1, arr2) is monotonic if:
  1. Both arrays have length n (same as nums).
  2. Both contain non-negative integers.
  3. arr1 is monotonically non-decreasing: arr1[0] <= arr1[1] <= ... <= arr1[n-1].
  4. arr2 is monotonically non-increasing: arr2[0] >= arr2[1] >= ... >= arr2[n-1].
  5. arr1[i] + arr2[i] == nums[i] for all 0 <= i <= n-1.

  The problem is solved using dynamic programming with prefix sums optimization.
  We count the number of valid `arr1` sequences, which uniquely determine `arr2`.
  The conditions on `arr1` are derived as:
  - 0 <= arr1[i] <= nums[i] for all i.
  - arr1[i+1] >= arr1[i] + max(0, nums[i+1] - nums[i]) for all 0 <= i <= n-2.

  Time complexity: O(n * K), where n is the length of nums and K is the maximum possible value in nums (K=50 based on constraints).
  Space complexity: O(K) for storing DP states and prefix sums.
  """
  def countOfPairs(self, nums: List[int]) -> int:
    """
    :param nums: List[int] The input array of positive integers. Length n >= 1. Values 1 <= nums[i] <= 50.
    :return: int The count of monotonic pairs modulo 10^9 + 7.
    """
    n = len(nums)
    
    # K represents the maximum possible value any nums[i] can take according to problem constraints.
    # The problem states 1 <= nums[i] <= 50. We use K=50 for array sizing.
    K = 50 
    
    # dp array stores counts for the previous step (initially step 0).
    # dp[j] represents the number of valid prefixes arr1[0...i-1] such that arr1[i-1] = j.
    # Size K+1 to accommodate indices from 0 to K.
    dp = [0] * (K + 1)
    
    # Base case for i = 0: Initialize DP for the first element arr1[0].
    # arr1[0] can be any integer j such that 0 <= j <= nums[0].
    # Each such choice forms a valid prefix sequence of length 1.
    for j in range(nums[0] + 1):
        # Since nums[0] <= K=50, j is always within the bounds of dp array [0, 50].
        dp[j] = 1
            
    # prefix_sum array stores prefix sums of the dp array for the previous step.
    # prefix_sum[x] = sum(dp[k] for k=0..x) % MOD. Used for O(1) range sum query.
    prefix_sum = [0] * (K + 1)
    prefix_sum[0] = dp[0]
    for j in range(1, K + 1):
        prefix_sum[j] = (prefix_sum[j-1] + dp[j]) % MOD
            
    # Iterate DP states from i = 1 to n-1 (computing states for index i based on i-1).
    for i in range(1, n):
        # new_dp array to compute DP states for the current step i.
        new_dp = [0] * (K + 1)
        
        # Calculate the constraint difference Li = max(0, nums[i] - nums[i-1]).
        # This comes from the combined monotonicity conditions:
        # arr1[i] must be at least arr1[i-1] + Li.
        Li = max(0, nums[i] - nums[i-1])
            
        # Compute new_dp[j] for all possible values j = arr1[i].
        # Constraint: 0 <= j <= nums[i].
        for j in range(nums[i] + 1):
            # Since nums[i] <= K=50, j is always within bounds [0, 50].
            
            # Determine the valid range for k = arr1[i-1].
            # k must satisfy:
            # 1. 0 <= k <= nums[i-1] (implicit from definition and arr2[i-1] >= 0)
            # 2. k <= j - Li (derived from arr1[i] >= arr1[i-1] + Li)
            # Combined upper bound for k: min(nums[i-1], j - Li).
            upper_k = min(nums[i-1], j - Li)
            
            # If upper_k < 0, the range [0, upper_k] is empty, sum is 0. No valid k exists.
            if upper_k >= 0:
                # The number of ways to reach state (i, j) is the sum of ways to reach valid previous states (i-1, k).
                # This sum is sum_{k=0}^{upper_k} dp[k] which equals prefix_sum[upper_k] from the previous step.
                # prefix_sum here refers to the prefix sums calculated based on dp state at step i-1.
                new_dp[j] = prefix_sum[upper_k]
            # else: new_dp[j] remains 0, which is correct as there are no valid previous states k.

        # Update dp array to hold the counts for the current step i.
        dp = new_dp 
            
        # Update prefix_sum array based on the new dp values for step i.
        # This updated prefix_sum array will be used in the next iteration (i+1).
        prefix_sum[0] = dp[0]
        for j in range(1, K + 1):
            prefix_sum[j] = (prefix_sum[j-1] + dp[j]) % MOD

    # After the loop finishes processing up to index n-1 (last element),
    # `dp` holds counts for the full arrays arr1[0...n-1] where arr1[n-1]=j.
    # `prefix_sum` holds the prefix sums for this final dp array (corresponding to step n-1).
    
    # The final answer is the total number of valid sequences arr1 of length n.
    # This is the sum of counts for all possible values of the last element arr1[n-1].
    # Constraint: 0 <= arr1[n-1] <= nums[n-1].
    # Total count = Sum_{j=0}^{nums[n-1]} dp[j] (where dp corresponds to step n-1).
    # This sum is readily available as prefix_sum[nums[n-1]].
    
    # Accessing prefix_sum[nums[n-1]] is safe because nums[n-1] <= K=50.
    final_ans = prefix_sum[nums[n-1]]
        
    return final_ans