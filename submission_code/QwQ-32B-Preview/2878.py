class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if k == 1:
            return all(nums[i] >= 0 for i in range(n))
        
        prefix_sum_ops = [0] * n
        prefix_sum_ops[0] = nums[0]
        
        for i in range(1, n):
            if i < k:
                prefix_sum_ops[i] = prefix_sum_ops[i - 1] + nums[i]
            else:
                prefix_sum_ops[i] = prefix_sum_ops[i - k] + nums[i]
            
            if i >= 1:
                ops_i = prefix_sum_ops[i] - prefix_sum_ops[i - 1]
                if ops_i < 0:
                    return False
            else:
                if prefix_sum_ops[0] < 0:
                    return False
        
        return True