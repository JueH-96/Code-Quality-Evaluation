from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of bit flip operations on array elements
        to make the bitwise XOR of all elements equal to k.

        The problem can be analyzed bit by bit independently.
        Let the initial XOR sum of all elements be `initial_xor`.
        We want the final XOR sum to be `k`.
        The operation is flipping a bit in any element. Flipping the i-th bit
        of an element changes the i-th bit of that element, and thus flips
        the i-th bit of the total XOR sum of the array.

        Consider the i-th bit position. Let the i-th bit of `initial_xor` be
        `b_i` and the i-th bit of `k` be `k_i`.
        If `b_i == k_i`, the i-th bit of the current XOR sum is already correct.
        To keep it correct, we need to perform an even number of flips at
        position i across all elements. The minimum such number is 0.
        If `b_i != k_i`, the i-th bit of the current XOR sum is incorrect.
        To make it correct, we need to perform an odd number of flips at
        position i across all elements. The minimum such number is 1 (by
        flipping the i-th bit of any single element).

        Since the decision for each bit position is independent, the total
        minimum number of operations is the sum of the minimum operations
        required for each bit position where `initial_xor` differs from `k`.

        The i-th bit of `initial_xor` differs from the i-th bit of `k` if and
        only if the i-th bit of `initial_xor ^ k` is 1.
        Therefore, the minimum number of operations is equal to the number
        of set bits (1s) in the binary representation of `initial_xor ^ k`.

        Args:
            nums: A 0-indexed integer array.
            k: A non-negative integer representing the target XOR sum.

        Returns:
            The minimum number of operations required.
        """
        # Calculate the initial XOR sum of all elements in nums.
        initial_xor = 0
        for num in nums:
            initial_xor ^= num

        # Calculate the difference between the initial XOR sum and the target k.
        # The set bits in this difference indicate the bit positions where
        # the initial XOR sum deviates from the target k.
        diff = initial_xor ^ k

        # Count the number of set bits (1s) in the difference.
        # Each set bit corresponds to a bit position that needs to be flipped
        # in the total XOR sum, requiring a minimum of 1 operation (flipping
        # that bit in one element).
        operations = 0
        # Use Brian Kernighan's algorithm for efficiency to count set bits.
        # This algorithm clears the least significant set bit in each iteration.
        while diff > 0:
            diff &= (diff - 1)
            operations += 1

        # Alternatively, using Python's built-in method (requires Python 3.10+):
        # operations = diff.bit_count()

        return operations