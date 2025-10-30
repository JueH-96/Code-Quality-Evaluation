class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Compute the XOR of all elements in nums.
        xor_val = 0
        for num in nums:
            xor_val ^= num
        
        # We want the final XOR to be k, so the total bit flips must
        # change xor_val into k. That difference is xor_val ^ k.
        diff = xor_val ^ k
        
        # Each bit that differs must be flipped exactly once
        # (flipping that bit in exactly one element is enough).
        # So the minimum number of flips is the number of set bits in diff.
        return diff.bit_count()