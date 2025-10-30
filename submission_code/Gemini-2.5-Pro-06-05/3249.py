from typing import List

class Solution:
  def minOperations(self, nums: List[int], k: int) -> int:
    """
    Calculates the minimum number of bit flips to make the XOR sum of an array equal to k.
    """
    
    # Step 1: Calculate the bitwise XOR sum of all elements in the array.
    # This gives us the current XOR state of the array.
    initial_xor_sum = 0
    for num in nums:
      initial_xor_sum ^= num
      
    # Step 2: Determine which bits need to be flipped.
    # To transform the initial_xor_sum into the target k, we need to flip
    # exactly the bits where they differ. The XOR operation `initial_xor_sum ^ k`
    # gives us a bitmask where each set bit (1) represents a position that
    # needs to be flipped.
    xor_difference = initial_xor_sum ^ k
    
    # Step 3: Count the number of set bits in the difference.
    # Each set bit in `xor_difference` corresponds to one bit that must be flipped
    # in the total XOR sum. Since one operation can flip one such bit, the
    # minimum number of operations is the number of set bits (population count).
    # In Python, `bin(n).count('1')` is a straightforward way to compute this.
    return bin(xor_difference).count('1')