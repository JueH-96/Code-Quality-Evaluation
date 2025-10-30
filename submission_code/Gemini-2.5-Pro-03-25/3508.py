import math # This import is not strictly necessary for the final solution but might be included by habit.

class Solution:
  """
  This class provides a solution to find the minimum number of bit changes (1->0) 
  to transform integer n into integer k.
  """
  def minChanges(self, n: int, k: int) -> int:
    """
    Calculates the minimum number of 1 -> 0 bit changes needed to make n equal to k.

    The operation allowed is to choose any bit in the binary representation of n 
    that is equal to 1 and change it to 0.

    Args:
      n: The starting positive integer (1 <= n <= 10^6).
      k: The target positive integer (1 <= k <= 10^6).

    Returns:
      The minimum number of changes required, or -1 if the transformation is impossible.
    """

    # Step 1: Check if the transformation is possible.
    # The only allowed operation is changing a '1' bit in n to '0'. We cannot change '0' to '1'.
    # This implies that if a bit position has a '1' in the target k, it must also have a '1' 
    # in the starting number n. Otherwise, we would need to change a '0' to a '1', which is forbidden.
    # We can check this condition using the bitwise AND operation. If `(n & k) == k`, 
    # it means every bit that is set (equal to 1) in `k` is also set in `n`.
    if (n & k) != k:
      # If the condition `(n & k) == k` is false, it means there exists at least one bit position 
      # where `k` has a '1' but `n` has a '0'. In this case, the transformation is impossible.
      return -1

    # Step 2: Calculate the number of changes needed.
    # If the transformation is possible (checked in Step 1), we need to determine how many 
    # '1' bits in `n` must be changed to '0' to match `k`.
    # A change is required at a bit position `i` if and only if `n` has a '1' at that position (`n[i] == 1`) 
    # and `k` has a '0' at that position (`k[i] == 0`).
    
    # Consider the bitwise XOR operation `n ^ k`. The result `n ^ k` has a '1' at bit positions 
    # where `n` and `k` differ, and '0' where they are the same.
    # The differing positions are where (`n[i] == 1` and `k[i] == 0`) or (`n[i] == 0` and `k[i] == 1`).
    
    # From Step 1, we know that the case (`n[i] == 0` and `k[i] == 1`) cannot occur if the transformation 
    # is possible. 
    # Therefore, if the transformation is possible, all the '1' bits in `n ^ k` must correspond 
    # exactly to the positions where `n[i] == 1` and `k[i] == 0`.
    
    # The number of changes needed is the count of these positions, which is equivalent to the 
    # number of set bits ('1's) in the result of `n ^ k`. This count is often called the 
    # population count or Hamming weight.
    
    difference_bits = n ^ k
    
    # Calculate the population count of `difference_bits`.
    # Python's `bin()` function converts an integer to its binary string representation 
    # (e.g., `bin(5)` returns `'0b101'`). 
    # The `count('1')` method of the string then counts the number of '1' characters.
    num_changes = bin(difference_bits).count('1')
    
    # Note: For Python 3.10 and later, a more direct way to get the popcount is:
    # num_changes = difference_bits.bit_count()

    return num_changes