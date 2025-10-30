import collections
from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations required to make the bitwise XOR 
        of all elements of the final array equal to k.

        An operation consists of choosing any element and flipping a bit in its 
        binary representation. Each such operation counts as one.

        The core idea relies on the property of bitwise XOR: flipping the i-th bit
        of any single number in the array will flip the i-th bit of the overall 
        XOR sum of the array, without affecting any other bit positions of the sum.
        Therefore, each bit position where the current XOR sum differs from the 
        target k requires exactly one operation to correct.

        Args:
            nums: A 0-indexed integer array.
            k: A non-negative integer, the target XOR sum.

        Returns:
            The minimum number of operations.
        """

        # Step 1: Calculate the initial bitwise XOR sum of all elements in nums.
        # This gives us the 'current' aggregated XOR state of the array.
        current_xor_sum = 0
        for num in nums:
            current_xor_sum ^= num
        
        # Step 2: Determine which specific bits in current_xor_sum need to be 
        # changed to match the corresponding bits in k.
        # The bitwise XOR of current_xor_sum and k (target_diff_xor) will have 
        # a '1' at precisely those bit positions where current_xor_sum's bit 
        # differs from k's bit. These are the bits that need to be flipped 
        # in the overall XOR sum.
        target_diff_xor = current_xor_sum ^ k
        
        # Step 3: The minimum number of operations is equal to the number of set bits (1s)
        # in target_diff_xor. Each set bit corresponds to one required bit flip 
        # in the overall XOR sum, which can be achieved by a single operation 
        # (flipping that specific bit in any chosen element of the array).
        operations_needed = 0
        
        # Count set bits using Brian Kernighan's algorithm.
        # This algorithm repeatedly clears the least significant set bit until 
        # the number becomes zero, counting each step.
        while target_diff_xor > 0:
            target_diff_xor &= (target_diff_xor - 1)
            operations_needed += 1
            
        return operations_needed