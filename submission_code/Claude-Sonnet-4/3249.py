class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate current XOR of all elements
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Find the difference between current XOR and target k
        diff = current_xor ^ k
        
        # Count the number of 1 bits in diff
        # Each 1 bit represents a position where we need to flip exactly one bit
        operations = 0
        while diff:
            operations += diff & 1
            diff >>= 1
        
        return operations