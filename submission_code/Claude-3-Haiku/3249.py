class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        xor = 0
        for num in nums:
            xor ^= num
        
        if xor == k:
            return 0
        
        # Count the number of bits that need to be flipped to make the XOR equal to k
        diff = xor ^ k
        
        # Count the number of operations required to make the XOR equal to k
        operations = 0
        for num in nums:
            if num & diff:
                operations += 1
        
        return operations