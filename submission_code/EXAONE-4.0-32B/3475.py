class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0 if all(x == 1 for x in nums) else -1
        
        ops = [0] * (n - 2)
        ops[0] = 1 - nums[0]
        
        if n >= 4:
            ops[1] = (1 - nums[1]) ^ ops[0]
        
        if n >= 5:
            for i in range(2, n - 2):
                ops[i] = (1 - nums[i]) ^ ops[i - 2] ^ ops[i - 1]
        
        parity_n2 = 0
        if n - 4 >= 0:
            parity_n2 ^= ops[n - 4]
        if n - 3 >= 0:
            parity_n2 ^= ops[n - 3]
        if nums[n - 2] ^ parity_n2 != 1:
            return -1
        
        parity_n1 = 0
        if n - 3 >= 0:
            parity_n1 = ops[n - 3]
        if nums[n - 1] ^ parity_n1 != 1:
            return -1
        
        return sum(ops)