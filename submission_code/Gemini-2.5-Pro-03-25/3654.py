import math
from typing import List

class Solution:
  """
  Solves the minimum array sum problem using dynamic programming.
  Finds the minimum possible sum of `nums` after applying at most `op1` divide-by-2 
  operations (rounding up) and at most `op2` subtract-k operations, with constraints 
  that each operation type can be applied at most once per index.
  """
  def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    """
    Calculates the minimum possible array sum using dynamic programming.

    Args:
      nums: The list of non-negative integers.
      k: The non-negative integer value to subtract in operation 2.
      op1: Maximum number of operation 1 (divide by 2, ceil) allowed.
      op2: Maximum number of operation 2 (subtract k) allowed.

    Returns:
      The minimum possible sum of the array after applying the operations.
    """
    
    n = len(nums)
    
    # dp[j][l] stores the minimum sum achievable using exactly j Operation 1 
    # and exactly l Operation 2 operations across the elements processed so far.
    # Initialize DP table with infinity. The dimensions accommodate 0 to op1 operations of type 1,
    # and 0 to op2 operations of type 2. Size is (op1+1) x (op2+1).
    # Using float('inf') for infinity representation ensures correct min comparisons.
    dp = [[float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
    
    # Base case: Before processing any elements (i.e., with an empty prefix), 
    # the sum is 0, achievable only with 0 operations of each type.
    dp[0][0] = 0
    
    # Process each number in the nums array one by one
    for i in range(n):
        num = nums[i]
        
        # Pre-calculate potential values after applying operations to the current number `num`.
        # This avoids redundant calculations inside the inner loops.
        
        # Operation 1 result: Divide `num` by 2, rounding up.
        # Integer division `(num + 1) // 2` correctly computes `ceil(num / 2)` for non-negative `num`.
        new1_val = (num + 1) // 2
        
        # Operation 2 condition and result: Subtract k, only possible if `num >= k`.
        cond2 = (num >= k)
        # Store result or float('inf') if the operation is impossible.
        new2_val = num - k if cond2 else float('inf') 
        
        # Combined Operation (Op1 then Op2) condition and result: 
        # First apply Op1 resulting in `new1_val`. Then apply Op2 if `new1_val >= k`.
        cond3 = (new1_val >= k)
        # Store result or float('inf') if the combined operation is impossible.
        new3_val = new1_val - k if cond3 else float('inf')
        
        # Iterate through the DP state space (number of Op1 used `j`, number of Op2 used `l`)
        # Iterate downwards (j from op1 down to 0, l from op2 down to 0).
        # This allows for an in-place update of the `dp` table because when calculating `dp[j][l]`,
        # the required values `dp[j-1][l]`, `dp[j][l-1]`, and `dp[j-1][l-1]` will still hold
        # the values from the previous iteration (before processing `num`).
        for j in range(op1, -1, -1):
            for l in range(op2, -1, -1):
                
                # Calculate the minimum possible sum for state (j, l) *after* considering `num`.
                # Initialize `current_min` to infinity; this will store the minimum possible value for dp[j][l]
                # based on the four choices for processing `num`.
                current_min = float('inf')

                # Possibility 1: No operation applied to `num`.
                # This transition comes from state (j, l) from the previous iteration.
                # The sum increases by the original value `num`.
                # `dp[j][l]` currently holds the value from the previous iteration because we haven't updated it yet in this step.
                if dp[j][l] != float('inf'):
                     current_min = min(current_min, dp[j][l] + num)

                # Possibility 2: Operation 1 applied to `num`.
                # This transition comes from state (j-1, l) from the previous iteration, consuming 1 Op1 budget.
                # The sum increases by `new1_val`. Requires `j >= 1`.
                # `dp[j-1][l]` holds the value from the previous iteration due to the downward loop order for `j`.
                if j >= 1 and dp[j-1][l] != float('inf'):
                     current_min = min(current_min, dp[j-1][l] + new1_val)

                # Possibility 3: Operation 2 applied to `num`.
                # This transition comes from state (j, l-1) from the previous iteration, consuming 1 Op2 budget.
                # Requires `num >= k` (checked by `cond2`). The sum increases by `new2_val`. Requires `l >= 1`.
                # `dp[j][l-1]` holds the value from the previous iteration due to the downward loop order for `l`.
                if l >= 1 and cond2 and dp[j][l-1] != float('inf'):
                     current_min = min(current_min, dp[j][l-1] + new2_val)

                # Possibility 4: Operation 1 then Operation 2 applied to `num`.
                # This transition comes from state (j-1, l-1) from the previous iteration, consuming 1 Op1 and 1 Op2 budget.
                # Requires `new1_val >= k` (checked by `cond3`). The sum increases by `new3_val`. Requires `j >= 1` and `l >= 1`.
                # `dp[j-1][l-1]` holds the value from the previous iteration due to downward loop orders.
                if j >= 1 and l >= 1 and cond3 and dp[j-1][l-1] != float('inf'):
                     current_min = min(current_min, dp[j-1][l-1] + new3_val)
                
                # Update dp[j][l] with the minimum sum found for this state after considering all possibilities for `num`.
                # This updated value reflects the state after processing element `i`.
                dp[j][l] = current_min

    # After processing all numbers in `nums`, the DP table `dp[j][l]` contains the minimum sum
    # achievable using exactly `j` Op1 and `l` Op2 operations.
    # The final answer is the minimum value across all possible states (j, l) such that 0 <= j <= op1 and 0 <= l <= op2.
    min_sum = float('inf')
    for j in range(op1 + 1):
        for l in range(op2 + 1):
            min_sum = min(min_sum, dp[j][l])
    
    # Under the given problem constraints (N >= 1, non-negative numbers), a finite minimum sum
    # should always be found (at least the sum without any operations is achievable).
    # If min_sum remains infinity, it indicates an issue, but this is unexpected.
    # The result should be an integer sum.
    
    # If min_sum is infinity, return 0 as a fallback (though it shouldn't happen).
    # Otherwise, cast the potentially float minimum sum to int.
    if min_sum == float('inf'):
       # This case signifies that no states were reachable, potentially due to empty input `nums`
       # or op1=op2=0 and some edge case. With N>=1, dp[0][0] will track the original sum.
       # If nums=[0], sum is 0. If nums=[large], sum is large. Always finite.
       # It might be possible if all initial states are inf except dp[0][0] = 0 and nums is empty.
       # However, N>=1 constraint prevents empty nums.
       # Defensive return 0 might hide bugs; assume problem constraints guarantee finite result.
       return 0 
       
    return int(min_sum)