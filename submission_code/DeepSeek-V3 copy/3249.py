class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate the current XOR of all elements in nums
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Determine the bits that need to be flipped
        target_xor = k
        xor_diff = current_xor ^ target_xor
        
        # Count the number of set bits in xor_diff
        operations = 0
        while xor_diff:
            operations += xor_diff & 1
            xor_diff >>= 1
        
        return operations