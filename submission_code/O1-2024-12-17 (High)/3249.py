class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Compute the XOR of all elements in nums
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # We want to transform current_xor into k by flipping bits.
        # Flipping a single bit in any element toggles exactly that bit in the overall XOR.
        # Therefore, the minimum number of flips is simply the number of differing bits
        # between current_xor and k, i.e., the popcount of current_xor ^ k.
        
        return bin(current_xor ^ k).count('1')