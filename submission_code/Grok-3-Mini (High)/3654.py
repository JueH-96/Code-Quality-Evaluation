import math
from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        NEG_INF = -1000000001
        dp = [[[NEG_INF for _ in range(op2 + 1)] for _ in range(op1 + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 0
        
        for i in range(1, n + 1):
            x = nums[i - 1]
            for curr_o1 in range(op1 + 1):
                for curr_o2 in range(op2 + 1):
                    # No operation
                    val_no_op = dp[i - 1][curr_o1][curr_o2] + 0
                    
                    # Operation 1 only
                    val_op1_only = NEG_INF
                    if curr_o1 >= 1:
                        red_op1 = x // 2
                        val_op1_only = dp[i - 1][curr_o1 - 1][curr_o2] + red_op1
                    
                    # Operation 2 only
                    val_op2_only = NEG_INF
                    if curr_o2 >= 1 and x >= k:
                        red_op2 = k
                        val_op2_only = dp[i - 1][curr_o1][curr_o2 - 1] + red_op2
                    
                    # Both operations
                    val_both = NEG_INF
                    if curr_o1 >= 1 and curr_o2 >= 1 and x >= k:
                        val1 = ((x - k) + 1) // 2  # ceil((x - k) / 2)
                        ceil_x_half = (x + 1) // 2  # ceil(x / 2)
                        if ceil_x_half >= k:
                            val2 = ceil_x_half - k
                            min_val = min(val1, val2)
                        else:
                            min_val = val1
                        red_both = x - min_val
                        val_both = dp[i - 1][curr_o1 - 1][curr_o2 - 1] + red_both
                    
                    # Set dp[i][curr_o1][curr_o2] to the maximum of all valid choices
                    dp[i][curr_o1][curr_o2] = max(val_no_op, val_op1_only, val_op2_only, val_both)
        
        # Find the maximum reduction possible with at most op1 and op2 operations
        max_red = max(dp[n][o1][o2] for o1 in range(op1 + 1) for o2 in range(op2 + 1))
        
        # Calculate the minimum sum
        sum_nums = sum(nums)
        min_sum = sum_nums - max_red
        return min_sum