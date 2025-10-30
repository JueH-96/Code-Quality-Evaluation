import math
from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations (bit flips) required
        to make the bitwise XOR of all elements in the array equal to k.

        The core idea is based on the property of the XOR operation.
        Flipping the p-th bit of any element `nums[i]` is equivalent to `nums[i] = nums[i] ^ (1 << p)`.
        When we consider the XOR sum of all elements in the array, `total_xor = nums[0] ^ nums[1] ^ ... ^ nums[n-1]`,
        flipping the p-th bit of `nums[i]` changes the `total_xor` as follows:
        `new_total_xor = nums[0] ^ ... ^ (nums[i] ^ (1 << p)) ^ ... ^ nums[n-1]`
        `new_total_xor = (nums[0] ^ ... ^ nums[i] ^ ... ^ nums[n-1]) ^ (1 << p)`
        `new_total_xor = total_xor ^ (1 << p)`
        This means that flipping the p-th bit of any element flips the p-th bit of the total XOR sum.

        Our goal is to make the final XOR sum equal to `k`. Let the initial XOR sum be `initial_xor`.
        We need to apply a series of bit flip operations such that the final XOR sum becomes `k`.
        Each operation flips one bit in the total XOR sum.
        The total change required in the XOR sum is `diff = initial_xor ^ k`.
        The bits that are set to 1 in `diff` are precisely the bits that need to be flipped in the total XOR sum
        to transform `initial_xor` into `k`.
        Since each operation flips exactly one bit position in the total XOR sum, the minimum number of operations
        required is equal to the number of bits that need to be flipped. This is equivalent to the number of
        set bits (1s) in the binary representation of `diff`.

        Args:
            nums: A list of 0-indexed integers.
            k: The target integer for the final XOR sum.

        Returns:
            The minimum number of operations required.
        """

        # 1. Calculate the initial XOR sum of all elements in the array.
        initial_xor = 0
        for num in nums:
            initial_xor ^= num

        # 2. Calculate the XOR difference between the initial XOR sum and the target k.
        # This value `diff` has a '1' at each bit position where `initial_xor` and `k` differ.
        # These are the bits in the total XOR sum that need to be flipped.
        diff = initial_xor ^ k

        # 3. Count the number of set bits (1s) in the binary representation of `diff`.
        # Each set bit indicates a required flip operation.
        # The total count of set bits is the minimum number of operations needed.
        operations = 0
        
        # Loop while there are bits left to check in diff
        # This loop iteratively checks the least significant bit and right-shifts `diff`.
        while diff > 0:
            # Check the least significant bit. If it's 1, increment operations count.
            # `diff & 1` evaluates to 1 if the last bit of `diff` is 1, and 0 otherwise.
            operations += diff & 1 
            
            # Right shift diff by 1 bit to process the next bit.
            # `diff >>= 1` is equivalent to `diff = diff // 2`.
            diff >>= 1
            
        # An alternative way to count set bits in Python is using `bin(diff).count('1')`.
        # For example: `operations = bin(diff).count('1')`
        # Both methods yield the same result. The loop is used here for fundamental clarity.

        return operations