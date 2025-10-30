class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        from functools import reduce
        from operator import xor
        
        # Compute the XOR of all elements in nums
        x = reduce(xor, nums, 0)
        
        # Determine how many bits differ between x and k
        diff = x ^ k
        
        # The answer is the number of set bits in diff (Hamming weight)
        return bin(diff).count('1')