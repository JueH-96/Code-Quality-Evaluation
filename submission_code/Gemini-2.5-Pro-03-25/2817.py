class Solution:
  """
  This class provides a solution to find the minimum cost to make all characters 
  of a binary string equal using prefix and suffix inversion operations.
  """
  def minimumCost(self, s: str) -> int:
    """
    Calculates the minimum cost to make all characters of the string 's' equal.

    The core idea is that the string needs to be made homogeneous (all '0's or all '1's).
    A string is homogeneous if and only if all adjacent characters are identical, i.e., s[i] == s[i+1] for all 0 <= i < n-1.
    If s[i] != s[i+1] for some index i, this represents a "boundary" that needs to be fixed.
    We need to apply operations to eliminate all such boundaries.

    There are two types of operations:
    1. Invert prefix s[0...i] with cost i + 1.
    2. Invert suffix s[i...n-1] with cost n - i.

    Let's analyze how these operations affect boundaries. A boundary i exists between s[i] and s[i+1].
    
    Operation 1 with index `k`: Inverts `s[0...k]`. Cost `k+1`.
    - This flips `s[k]`. 
    - If `k+1 < n`, `s[k+1]` is not flipped. 
    - Thus, this operation changes the state of boundary `k` (between `s[k]` and `s[k+1]`).
    - For any boundary `j < k`, both `s[j]` and `s[j+1]` are flipped. So `s[j] == s[j+1]` remains true/false, and `s[j] != s[j+1]` remains true/false. Boundary `j` state is unchanged.
    - Conclusion: Operation 1 with index `k` flips only boundary `k`. Cost `k+1`.

    Operation 2 with index `k`: Inverts `s[k...n-1]`. Cost `n-k`.
    - This flips `s[k]`.
    - If `k > 0`, `s[k-1]` is not flipped.
    - Thus, this operation changes the state of boundary `k-1` (between `s[k-1]` and `s[k]`).
    - For any boundary `j >= k`, both `s[j]` and `s[j+1]` are flipped. Boundary `j` state is unchanged.
    - For any boundary `j < k-1`, neither `s[j]` nor `s[j+1]` are flipped. Boundary `j` state is unchanged.
    - Conclusion: Operation 2 with index `k` flips only boundary `k-1`. Cost `n-k`.

    To fix a boundary `i` (where `s[i] != s[i+1]`), we need to apply an operation that flips this boundary. We have two choices:
    1. Use Operation 1 with index `i`. This flips boundary `i`. Cost is `i + 1`.
    2. Use Operation 2 with index `i + 1`. This flips boundary `(i+1)-1 = i`. Cost is `n - (i + 1)`.

    Both operations target boundary `i` specifically and do not affect any other boundaries.
    Therefore, for each boundary `i` where `s[i] != s[i+1]`, we must apply one of these two operations. To minimize the total cost, we should choose the cheaper operation for each such boundary independently.
    The minimum cost to fix boundary `i` is `min(i + 1, n - (i + 1))`.

    The total minimum cost is the sum of these minimum costs over all boundaries `i` where `s[i] != s[i+1]`.

    Args:
      s: The input binary string.

    Returns:
      The minimum cost to make all characters of the string equal.
    """
    n = len(s)
    # If the string has 0 or 1 character, it's already homogeneous. Cost is 0.
    if n <= 1:
        return 0
    
    total_cost = 0
    # Iterate through all adjacent pairs of characters.
    # There are n-1 potential boundaries, indexed from 0 to n-2.
    # Boundary i is between s[i] and s[i+1].
    for i in range(n - 1):
        # If adjacent characters are different, this boundary needs to be flipped.
        if s[i] != s[i+1]:
            # Calculate the cost of the two options to flip boundary i:
            # Option 1: Use Operation 1 with index i. Cost: i + 1.
            cost_op1 = i + 1
            
            # Option 2: Use Operation 2 with index i + 1. Cost: n - (i + 1).
            cost_op2 = n - (i + 1)
            
            # Add the minimum of the two costs to the total cost.
            total_cost += min(cost_op1, cost_op2)
            
    # The total minimum cost is the sum accumulated across all boundaries that needed fixing.
    return total_cost