import math
from typing import List

class Solution:
  def minOperations(self, nums: List[int], k: int) -> int:
    """
    Calculates the minimum number of operations to make the bitwise XOR of all elements
    of the array equal to k. An operation consists of flipping a bit in an element.

    Args:
      nums: A 0-indexed integer array.
      k: An integer target for the XOR sum. (Constraints state 0 <= k <= 10^6)

    Returns:
      The minimum number of operations.
    """
    
    # Calculate the initial XOR sum of all elements in nums.
    # current_xor_sum will hold the XOR sum of all elements.
    # Initialize with 0 because 0 XOR x = x.
    current_xor_sum = 0
    for num in nums:
      current_xor_sum ^= num
      
    # Let S_initial be current_xor_sum.
    # We want the final XOR sum of the array to be k. Let this be S_final.
    
    # Consider the effect of one operation:
    # If we choose an element nums[idx] and flip its j-th bit,
    # nums[idx] changes to nums[idx] XOR (1 << j).
    # The total XOR sum of the array changes from S_initial to S_initial XOR (1 << j).
    # So, each operation allows us to flip one specific bit (at position j) 
    # in the total XOR sum S_initial. This costs 1 operation.
    
    # We need to transform S_initial to S_final (which is k) using the minimum number of operations.
    # The bits that need to be flipped are precisely those where S_initial and S_final differ.
    # The number of such differing bits is the Hamming distance between S_initial and S_final.
    # Hamming_distance(A, B) is equal to the population count (number of set bits) 
    # of (A XOR B).
    
    # Calculate the value representing the bit differences between current_xor_sum and k.
    # This value, diff_val, will have a '1' at bit position j if the j-th bit of 
    # current_xor_sum differs from the j-th bit of k, and '0' otherwise.
    diff_val = current_xor_sum ^ k
    
    # Count the number of set bits (1s) in diff_val.
    # This count is the minimum number of operations required, as each operation
    # can correct one differing bit.
    # Python's bin(x) converts a non-negative integer x to its binary string representation
    # (e.g., bin(5) is '0b101'). We can then count the occurrences of '1'.
    operations = bin(diff_val).count('1')
      
    return operations