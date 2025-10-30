class Solution:
    def minOperations(self, nums: list[int], k: int) -> int:
        # Calculate the XOR of all elements in nums
        xor_all = 0
        for num in nums:
            xor_all ^= num
        
        # If the XOR of all elements is already k, no operations are needed
        if xor_all == k:
            return 0
        
        # Calculate the XOR of k and xor_all to find the bits that need to be flipped
        xor_diff = xor_all ^ k
        
        # Count the number of bits set in xor_diff
        # This will give us the minimum number of operations required
        operations = 0
        while xor_diff:
            operations += xor_diff & 1
            xor_diff >>= 1
        
        return operations