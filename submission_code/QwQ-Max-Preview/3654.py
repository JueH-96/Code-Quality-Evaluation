from typing import List
import sys

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        original_sum = sum(nums)
        max_op1 = op1
        max_op2 = op2
        
        # Initialize DP with -infinity except for the starting state
        dp = [[-sys.maxsize for _ in range(max_op2 + 1)] for __ in range(max_op1 + 1)]
        dp[max_op1][max_op2] = 0  # initial state with all operations available
        
        for num in nums:
            # Generate all possible options for the current number
            options = set()
            # Option a: do nothing
            options.add((0, 0, 0))
            # Option b: apply op1 only
            new_num_b = (num + 1) // 2
            red_b = num - new_num_b
            options.add((red_b, 1, 0))
            # Option c: apply op2 only if possible
            if num >= k:
                red_c = k
                options.add((red_c, 0, 1))
            # Option d: apply op1 then op2 if possible
            new_after_op1 = (num + 1) // 2
            if new_after_op1 >= k:
                red_d = num - (new_after_op1 - k)
                options.add((red_d, 1, 1))
            # Option e: apply op2 then op1 if possible
            if num >= k:
                after_op2_e = num - k
                new_after_op1_e = (after_op2_e + 1) // 2
                red_e = num - new_after_op1_e
                options.add((red_e, 1, 1))
            
            # Convert to list to process
            options = list(options)
            
            # Create temporary DP for the new state
            tmp_dp = [[-sys.maxsize for _ in range(max_op2 + 1)] for __ in range(max_op1 + 1)]
            
            for available_op1 in range(max_op1, -1, -1):
                for available_op2 in range(max_op2, -1, -1):
                    if dp[available_op1][available_op2] == -sys.maxsize:
                        continue
                    current_red = dp[available_op1][available_op2]
                    for (red, used_op1, used_op2) in options:
                        if available_op1 >= used_op1 and available_op2 >= used_op2:
                            new_available_op1 = available_op1 - used_op1
                            new_available_op2 = available_op2 - used_op2
                            if tmp_dp[new_available_op1][new_available_op2] < current_red + red:
                                tmp_dp[new_available_op1][new_available_op2] = current_red + red
            
            # Update dp to tmp_dp for the next iteration
            dp = tmp_dp
        
        # Find the maximum possible reduction
        max_reduction = max(max(row) for row in dp)
        return original_sum - max_reduction