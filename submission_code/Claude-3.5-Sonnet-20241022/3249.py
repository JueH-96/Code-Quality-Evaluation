class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Get XOR of all elements in nums
        curr_xor = 0
        for num in nums:
            curr_xor ^= num
            
        # If current XOR equals target k, no operations needed
        if curr_xor == k:
            return 0
            
        # Count differing bits between current XOR and target k
        operations = 0
        curr_xor_bits = bin(curr_xor)[2:] # Remove '0b' prefix
        k_bits = bin(k)[2:]
        
        # Make both binary strings same length by padding with zeros
        max_len = max(len(curr_xor_bits), len(k_bits))
        curr_xor_bits = curr_xor_bits.zfill(max_len)
        k_bits = k_bits.zfill(max_len)
        
        # Count positions where bits differ
        for i in range(max_len):
            if curr_xor_bits[i] != k_bits[i]:
                operations += 1
                
        # For each differing bit, we need to flip exactly one bit in one number
        # to change the XOR at that position
        return operations