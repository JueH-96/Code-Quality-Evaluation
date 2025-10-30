class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        initial_sum = sum(nums)
        dp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for num in nums:
            reduction1 = num - ((num + 1) // 2)
            reduction2 = k if num >= k else 0
            
            reduction_a = 0
            op2_result = num - k
            if op2_result >= 0:
                op1_after_op2 = (op2_result + 1) // 2
                reduction_a = op2_result - op1_after_op2
            
            reduction_b = 0
            op1_result = (num + 1) // 2
            if op1_result >= 0:
                op2_after_op1 = op1_result - k
                if op2_after_op1 >= 0:
                    reduction_b = op1_result - op2_after_op1
            
            option3_reduction = max(reduction_a, reduction_b)
            options = [
                (0, 0, 0),
                (1, 0, reduction1),
                (0, 1, reduction2),
                (1, 1, option3_reduction)
            ]
            
            dp_temp = [[-float('inf')] * (op2 + 1) for _ in range(op1 + 1)]
            for i in range(op1 + 1):
                for j in range(op2 + 1):
                    current_reduction = dp[i][j]
                    if current_reduction == -float('inf'):
                        continue
                    for op in options:
                        op1_used, op2_used, reduction = op
                        new_i = i + op1_used
                        new_j = j + op2_used
                        if new_i > op1 or new_j > op2:
                            continue
                        if current_reduction + reduction > dp_temp[new_i][new_j]:
                            dp_temp[new_i][new_j] = current_reduction + reduction
            dp = dp_temp
        
        max_reduction = max(dp[i][j] for i in range(op1 + 1) for j in range(op2 + 1))
        minimal_sum = initial_sum - max_reduction
        return minimal_sum