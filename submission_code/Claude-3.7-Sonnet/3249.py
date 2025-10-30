class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate the current XOR of all elements
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Find which bits differ between current_xor and k
        diff = current_xor ^ k
        
        # Count the number of bits that need to be flipped
        # (i.e., the number of 1's in the binary representation of diff)
        return bin(diff).count('1')