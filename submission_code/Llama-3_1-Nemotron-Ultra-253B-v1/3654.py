from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        original_sum = sum(nums)
        dp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for num in nums:
            options = self.generate_options(num, k)
            for o1, o2, red in options:
                for i in range(op1, o1 - 1, -1):
                    for j in range(op2, o2 - 1, -1):
                        if dp[i - o1][j - o2] + red > dp[i][j]:
                            dp[i][j] = dp[i - o1][j - o2] + red
        
        max_reduction = max(max(row) for row in dp)
        return original_sum - max_reduction
    
    def generate_options(self, x: int, k: int) -> List[tuple]:
        options = []
        options.append((0, 0, 0))
        
        reduction1 = x - ((x + 1) // 2)
        options.append((1, 0, reduction1))
        
        if x >= k:
            options.append((0, 1, k))
        
        max_reduction = 0
        possible = False
        
        after_op1 = (x + 1) // 2
        if after_op1 >= k:
            reduction = x - (after_op1 - k)
            possible = True
            max_reduction = reduction
        
        if x >= k:
            after_op2 = x - k
            after_op1_after_op2 = (after_op2 + 1) // 2
            reduction = x - after_op1_after_op2
            if reduction > max_reduction:
                max_reduction = reduction
            possible = True
        
        if possible:
            options.append((1, 1, max_reduction))
        
        return options