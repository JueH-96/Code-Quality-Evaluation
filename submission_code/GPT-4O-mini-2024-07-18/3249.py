class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        from functools import reduce
        
        # Calculate the XOR of all elements in nums
        current_xor = reduce(lambda x, y: x ^ y, nums)
        
        # If current XOR is already equal to k, no operations are needed
        if current_xor == k:
            return 0
        
        # Calculate the XOR that we need to achieve
        target_xor = current_xor ^ k
        
        # Count the number of bits that differ between current_xor and k
        # This is equivalent to counting the number of 1s in target_xor
        operations = bin(target_xor).count('1')
        
        return operations