class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        
        # dp[i][j][l] = minimum sum for first i elements using j op1s and l op2s
        dp = [[[float('inf')] * (op2 + 1) for _ in range(op1 + 1)] for _ in range(n + 1)]
        dp[0][0][0] = 0
        
        for i in range(n):
            for j in range(op1 + 1):
                for l in range(op2 + 1):
                    if dp[i][j][l] == float('inf'):
                        continue
                    
                    # No operations on nums[i]
                    dp[i+1][j][l] = min(dp[i+1][j][l], dp[i][j][l] + nums[i])
                    
                    # Operation 1 only
                    if j < op1:
                        val1 = (nums[i] + 1) // 2
                        dp[i+1][j+1][l] = min(dp[i+1][j+1][l], dp[i][j][l] + val1)
                    
                    # Operation 2 only
                    if l < op2 and nums[i] >= k:
                        val2 = nums[i] - k
                        dp[i+1][j][l+1] = min(dp[i+1][j][l+1], dp[i][j][l] + val2)
                    
                    # Both operations
                    if j < op1 and l < op2:
                        val_both_options = []
                        
                        # Operation 1 then operation 2
                        temp = (nums[i] + 1) // 2
                        if temp >= k:
                            val_both_options.append(temp - k)
                        
                        # Operation 2 then operation 1
                        if nums[i] >= k:
                            val_both_options.append(((nums[i] - k) + 1) // 2)
                        
                        if val_both_options:
                            val_both = min(val_both_options)
                            dp[i+1][j+1][l+1] = min(dp[i+1][j+1][l+1], dp[i][j][l] + val_both)
        
        # Find the minimum among all possible states
        result = float('inf')
        for j in range(op1 + 1):
            for l in range(op2 + 1):
                result = min(result, dp[n][j][l])
        
        return result