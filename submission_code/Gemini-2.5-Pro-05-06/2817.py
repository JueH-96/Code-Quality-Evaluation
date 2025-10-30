import math

class Solution:
  def minimumCost(self, s: str) -> int:
    n = len(s)
    
    # If n is 1, the string already has all characters equal. Cost is 0.
    # The loop `range(n - 1)` correctly handles this:
    # for n=1, range(0) is empty, and total_cost remains 0.
    
    total_cost = 0
    
    # We iterate through all adjacent character pairs (s[k], s[k+1]).
    # The loop for k goes from 0 to n-2.
    for k in range(n - 1):
      # If characters at k and k+1 are different, a "break" exists.
      # This break must be fixed to ensure all characters in the string become equal.
      if s[k] != s[k+1]:
        # To fix the break s[k] != s[k+1] (i.e., to make s'[k] == s'[k+1] in the final string),
        # we must choose an operation that affects s[k] and s[k+1] differently.
        
        # Option 1: Use a prefix flip ending at k.
        # This operation is "invert all characters from index 0 to index k".
        # It flips s[k] but does not flip s[k+1].
        # The cost of this operation is (k + 1).
        cost_prefix_op = k + 1
        
        # Option 2: Use a suffix flip starting at k+1.
        # This operation is "invert all characters from index k+1 to index n-1".
        # It flips s[k+1] but does not flip s[k].
        # The cost of this operation is n - (k + 1). (Here, `i` from the problem spec is `k+1`)
        cost_suffix_op = n - (k + 1)
        
        # We choose the cheaper of these two options to fix the break at k.
        # The critical insight is that an operation chosen to fix the break at index k
        # (either a prefix flip ending at k, or a suffix flip starting at k+1)
        # does not alter the equality status (s[j] == s[j+1] or s[j] != s[j+1])
        # of any OTHER pair (s[j], s[j+1]) where j != k.
        # This means choices for different breaks are independent.
        #
        # Therefore, the total minimum cost is the sum of minimum costs for fixing each individual break.
        total_cost += min(cost_prefix_op, cost_suffix_op)
        
    return total_cost