class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate XOR of all elements
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Find the XOR between current result and target k
        # This gives us the bits that need to be flipped
        diff = current_xor ^ k
        
        # Count the number of 1s in the binary representation
        # This is the minimum number of bit flips needed
        count = 0
        while diff > 0:
            count += diff & 1
            diff >>= 1
        
        return count