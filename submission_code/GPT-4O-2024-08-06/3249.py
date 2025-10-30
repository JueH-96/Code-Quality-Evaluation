class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate the XOR of all elements in nums
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Calculate the target XOR we need to achieve
        target_xor = current_xor ^ k
        
        # If the current XOR is already equal to k, no operations are needed
        if target_xor == 0:
            return 0
        
        # Count the number of bits that differ between current_xor and k
        operations = 0
        while target_xor > 0:
            # Count the number of 1s in the binary representation of target_xor
            operations += target_xor & 1
            target_xor >>= 1
        
        return operations