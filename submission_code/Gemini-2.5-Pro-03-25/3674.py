import sys 
from typing import List

# It's good practice to increase recursion depth if using recursive structures
# like segment trees, though this solution is iterative.
# sys.setrecursionlimit(200000) 

class Solution:
  """
  Calculates the number of subarrays of `nums` that can be made non-decreasing
  using at most `k` operations. Each operation increments an element by 1.
  The minimum number of operations required for a subarray `nums[i..j]` is
  calculated as `sum_{p=i..j} (b_p - nums[p])`, where `b_i = nums[i]` and
  `b_p = max(b_{p-1}, nums[p])` for `p > i`. This can be shown to be equivalent to
  `sum_{p=i..j} (M(i, p) - nums[p])` where `M(i, p) = max(nums[i], ..., nums[p])`.

  This implementation uses an O(N^2) approach by iterating through all possible
  subarrays `nums[i..j]` and calculating the minimum cost for each. It checks if
  the cost is within the budget `k`. While correct, this approach is likely too
  slow for the given constraints (N <= 10^5) and may result in Time Limit Exceeded.
  An optimized solution would typically require O(N log N) or O(N) complexity, possibly
  using techniques like two pointers combined with data structures (e.g., segment trees
  or monotonic stacks) to efficiently update and query costs.
  """
  def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
    """
    Args:
        nums: A list of integers representing the array.
        k: The maximum number of operations allowed per subarray.

    Returns:
        The total count of subarrays that can be made non-decreasing within k operations.
    """
    n = len(nums)
    ans = 0 # Initialize the count of valid subarrays

    # Iterate through all possible start indices `i` of subarrays
    for i in range(n):
        # Initialize state for subarrays starting at `i`
        
        # current_max_val tracks M(i, p), the maximum value in nums[i..p].
        # Initialized for p=i. M(i, i) = nums[i].
        current_max_val = nums[i] 
        
        # current_sum_max tracks S_M(i, p) = sum_{x=i..p} M(i, x).
        # Initialized for p=i. S_M(i, i) = M(i, i) = nums[i].
        current_sum_max = nums[i] 
        
        # subarray_sum tracks S_N(i, p) = sum_{x=i..p} nums[x].
        # Initialized for p=i. S_N(i, i) = nums[i].
        subarray_sum = nums[i]
        
        # Calculate the cost for the single-element subarray [i, i].
        # The minimum cost to make a single-element array non-decreasing is 0.
        # cost(i, i) = S_M(i, i) - S_N(i, i) = nums[i] - nums[i] = 0
        cost = 0 
        
        # A subarray of length 1 always has cost 0. Since the problem constraints state k >= 1,
        # any single-element subarray is always valid.
        ans += 1 

        # Extend the subarray to the right, checking indices j from i+1 to n-1
        for j in range(i + 1, n):
            # Update the maximum value encountered so far in the subarray [i, j].
            # M(i, j) = max(M(i, j-1), nums[j]), where M(i, j-1) is stored in current_max_val.
            current_max_val = max(current_max_val, nums[j]) 
            
            # Update the sum of maximums S_M(i, j).
            # S_M(i, j) = S_M(i, j-1) + M(i, j). S_M(i, j-1) is the previous value of current_sum_max.
            current_sum_max += current_max_val
            
            # Update the sum of the subarray elements S_N(i, j).
            # S_N(i, j) = S_N(i, j-1) + nums[j]. S_N(i, j-1) is the previous value of subarray_sum.
            subarray_sum += nums[j] 
            
            # Calculate the cost for the subarray [i, j].
            # cost(i, j) = S_M(i, j) - S_N(i, j).
            cost = current_sum_max - subarray_sum
            
            # Check if the cost is within the allowed budget k.
            if cost <= k:
                # If the cost is within budget, increment the count of valid subarrays.
                ans += 1
            else:
                # Optimization: The cost function cost(i, j) is non-decreasing with respect to j for a fixed i.
                # This is because M(i, p) is non-decreasing in p, so S_M(i, j) grows at least as fast as S_N(i, j).
                # If the cost exceeds k for subarray [i, j], it will also exceed k for any larger subarray [i, p] where p > j.
                # Therefore, we can stop extending the subarray starting at index i and move to the next starting index i+1.
                break
                
    # Return the total count of valid subarrays found.
    return ans