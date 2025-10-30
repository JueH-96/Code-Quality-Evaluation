class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Step 1: Calculate the XOR of all elements in nums
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # Step 2: Calculate the XOR of current_xor and k
        diff = current_xor ^ k
        
        # Step 3: Count the number of set bits in diff
        return diff.bit_count()