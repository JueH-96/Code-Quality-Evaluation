from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        initial_sum = sum(nums)
        options_list = []
        
        for x in nums:
            opts = []
            # Option 0: do nothing
            opts.append((0, 0, 0))
            
            # Option 1: apply op1
            r1 = x - (x + 1) // 2
            opts.append((1, 0, r1))
            
            if x >= k:
                # Option 2: apply op2
                r2 = k
                opts.append((0, 1, r2))
                
                # Option 3: apply both in best order
                v1 = (x + 1) // 2
                option3a = r1 + k if v1 >= k else -float('inf')
                new_val_b = (x - k + 1) // 2
                option3b = x - new_val_b
                max_option3 = max(option3a, option3b)
                opts.append((1, 1, max_option3))
            
            options_list.append(opts)
        
        # Initialize DP table
        INF = float('inf')
        dp = [[-INF for _ in range(op2 + 1)] for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for opts in options_list:
            temp_dp = [row.copy() for row in dp]
            for a in range(op1 + 1):
                for b in range(op2 + 1):
                    if dp[a][b] == -INF:
                        continue
                    for (da, db, dr) in opts:
                        new_a = a + da
                        new_b = b + db
                        if new_a <= op1 and new_b <= op2:
                            if temp_dp[new_a][new_b] < dp[a][b] + dr:
                                temp_dp[new_a][new_b] = dp[a][b] + dr
            dp = temp_dp
        
        max_reduction = 0
        for a in range(op1 + 1):
            for b in range(op2 + 1):
                if dp[a][b] > max_reduction:
                    max_reduction = dp[a][b]
        
        return initial_sum - max_reduction