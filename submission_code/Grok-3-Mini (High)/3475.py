class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        length_ops = n - 2
        x_ops = [0] * length_ops
        x_ops[0] = 1 - nums[0]
        if length_ops > 1:
            x_ops[1] = (1 - nums[1]) ^ x_ops[0]
            for k in range(2, length_ops):
                x_ops[k] = (1 - nums[k]) ^ x_ops[k - 2] ^ x_ops[k - 1]
        
        # Check position n-1
        sum_pos_n1 = x_ops[-1]
        req_n1 = 1 - nums[n - 1]
        if sum_pos_n1 != req_n1:
            return -1
        
        # Check position n-2
        low = max(0, n - 4)
        high = n - 3  # inclusive
        sum_pos_n2 = sum(x_ops[low : high + 1])
        req_n2 = 1 - nums[n - 2]
        if sum_pos_n2 != req_n2:
            return -1
        
        # If both checks pass, return the sum of operations
        return sum(x_ops)