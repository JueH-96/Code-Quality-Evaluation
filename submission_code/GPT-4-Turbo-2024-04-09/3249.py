class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # If the current XOR is already k, no operations are needed
        if current_xor == k:
            return 0
        
        # Calculate the XOR difference that we need to achieve
        xor_diff = current_xor ^ k
        
        # We need to find the minimum number of bit flips to turn current_xor into k
        # This is equivalent to counting the number of 1's in xor_diff (since each 1 represents a bit flip)
        return bin(xor_diff).count('1')