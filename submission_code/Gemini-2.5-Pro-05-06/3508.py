class Solution:
  def minChanges(self, n: int, k: int) -> int:
    """
    Calculates the minimum number of changes to make n equal to k.
    A change consists of choosing a bit in n that is 1 and changing it to 0.
    If impossible, returns -1.

    Args:
      n: The initial positive integer.
      k: The target positive integer.

    Returns:
      The minimum number of changes, or -1 if impossible.
    """

    # Step 1: Check for impossibility.
    # If k has a '1' bit where n has a '0' bit, it's impossible
    # because we can only change 1s to 0s, not 0s to 1s.
    # This is true if (n & k) is not equal to k.
    if (n & k) != k:
      return -1
    
    # Step 2: Calculate the number of changes.
    # If possible, changes are needed where n has a '1' but k has a '0'.
    # The expression (n ^ k) will have a '1' at bit positions where n and k differ.
    # Since we've already established that there are no positions where n_i=0 and k_i=1,
    # all differing bits must be of the form n_i=1 and k_i=0.
    # So, the number of changes is the population count of (n ^ k).
    
    difference_bits = n ^ k
    
    # Calculate population count (number of set bits).
    # bin(number).count('1') is a standard way to do this in Python.
    # int.bit_count() is available in Python 3.10+ and can be more efficient.
    # For typical contest environments, either is fine. We use bin().count('1') for wider compatibility.
    return bin(difference_bits).count('1')