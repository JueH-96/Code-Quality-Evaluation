class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Calculate XOR of all elements
        xor = 0
        for num in nums:
            xor ^= num
        
        # Find bits that differ between current XOR and target k
        # Count the number of 1s in the XOR (Hamming distance)
        return bin(xor ^ k).count('1')