class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        # Define DP array: dp[i][j][l] = minimum sum after processing the first i elements
        # using j operations of type 1 and l operations of type 2
        dp = [[[float('inf') for _ in range(op2 + 1)] for _ in range(op1 + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 0  # Base case: no elements processed, no operations used
        
        for i in range(1, n + 1):
            num = nums[i - 1]  # Current element
            
            for j in range(op1 + 1):
                for l in range(op2 + 1):
                    # Option 1: Do nothing to this element
                    if dp[i-1][j][l] != float('inf'):
                        dp[i][j][l] = min(dp[i][j][l], dp[i-1][j][l] + num)
                    
                    # Option 2: Apply only Operation 1
                    if j > 0 and dp[i-1][j-1][l] != float('inf'):
                        dp[i][j][l] = min(dp[i][j][l], dp[i-1][j-1][l] + (num + 1) // 2)
                    
                    # Option 3: Apply only Operation 2
                    if l > 0 and num >= k and dp[i-1][j][l-1] != float('inf'):
                        dp[i][j][l] = min(dp[i][j][l], dp[i-1][j][l-1] + (num - k))
                    
                    # Option 4: Apply both operations (find the best sequence)
                    if j > 0 and l > 0 and num >= k and dp[i-1][j-1][l-1] != float('inf'):
                        # Sequence: Op1 -> Op2
                        temp_val = (num + 1) // 2  # After Op1
                        new_val_op1_op2 = max(0, temp_val - k) if temp_val >= k else temp_val
                        
                        # Sequence: Op2 -> Op1
                        temp_val = num - k  # After Op2
                        new_val_op2_op1 = (temp_val + 1) // 2  # After Op1
                        
                        # Choose the best sequence
                        min_val_after_both_ops = min(new_val_op1_op2, new_val_op2_op1)
                        
                        dp[i][j][l] = min(dp[i][j][l], dp[i-1][j-1][l-1] + min_val_after_both_ops)
        
        return dp[n][op1][op2]