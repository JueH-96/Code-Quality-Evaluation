class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        xor_all = 0
        for num in nums:
            xor_all ^= num

        # If the XOR of all elements is already equal to k, no operations are needed
        if xor_all == k:
            return 0

        # To make the XOR of all elements equal to k, we need to flip bits such that
        # the XOR of the final array is k. The number of bits that need to be flipped
        # is the number of bits that differ between xor_all and k.
        diff = xor_all ^ k
        # Count the number of 1s in the binary representation of diff
        return bin(diff).count('1')