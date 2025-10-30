class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        # Calculate the reduction for each element with different operation combinations
        reductions = []
        
        for i in range(n):
            num = nums[i]
            
            # No operation
            no_op = 0
            
            # Only operation 1 (divide by 2, round up)
            op1_only = num - ((num + 1) // 2)
            
            # Only operation 2 (subtract k if possible)
            op2_only = k if num >= k else 0
            
            # Both operations - we need to consider the order
            # Option 1: Apply op1 first, then op2
            after_op1 = (num + 1) // 2
            both_op1_first = op1_only + (k if after_op1 >= k else 0)
            
            # Option 2: Apply op2 first, then op1
            after_op2 = num - k if num >= k else num
            both_op2_first = (k if num >= k else 0) + (after_op2 - ((after_op2 + 1) // 2))
            
            # Choose the better order for both operations
            both_ops = max(both_op1_first, both_op2_first)
            
            reductions.append((op1_only, op2_only, both_ops, i))
        
        # Use dynamic programming to find the optimal allocation
        # dp[i][j][l] = maximum reduction using first i elements with j op1s and l op2s used
        dp = {}
        
        def solve(idx, op1_left, op2_left):
            if idx == n:
                return 0
            
            if (idx, op1_left, op2_left) in dp:
                return dp[(idx, op1_left, op2_left)]
            
            op1_only, op2_only, both_ops, _ = reductions[idx]
            
            # Option 1: Don't use any operation on this element
            result = solve(idx + 1, op1_left, op2_left)
            
            # Option 2: Use only op1
            if op1_left > 0:
                result = max(result, op1_only + solve(idx + 1, op1_left - 1, op2_left))
            
            # Option 3: Use only op2
            if op2_left > 0:
                result = max(result, op2_only + solve(idx + 1, op1_left, op2_left - 1))
            
            # Option 4: Use both operations
            if op1_left > 0 and op2_left > 0:
                result = max(result, both_ops + solve(idx + 1, op1_left - 1, op2_left - 1))
            
            dp[(idx, op1_left, op2_left)] = result
            return result
        
        max_reduction = solve(0, op1, op2)
        return sum(nums) - max_reduction