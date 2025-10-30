class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        INF = 10**18
        max_op1 = op1
        max_op2 = op2
        
        # Initialize DP table: dp[a][b] is the minimal sum with a op1 left and b op2 left
        dp = [[INF] * (max_op2 + 1) for _ in range(max_op1 + 1)]
        dp[max_op1][max_op2] = 0  # Initial state: no elements processed
        
        for x in nums:
            new_dp = [[INF] * (max_op2 + 1) for _ in range(max_op1 + 1)]
            for a_prev in range(max_op1 + 1):
                for b_prev in range(max_op2 + 1):
                    if dp[a_prev][b_prev] == INF:
                        continue
                    current_sum = dp[a_prev][b_prev]
                    
                    # Option 1: Do nothing
                    new_a, new_b = a_prev, b_prev
                    cost = x
                    if new_dp[new_a][new_b] > current_sum + cost:
                        new_dp[new_a][new_b] = current_sum + cost
                    
                    # Option 2: Apply op1
                    if a_prev > 0:
                        cost = (x + 1) // 2
                        new_a, new_b = a_prev - 1, b_prev
                        if new_dp[new_a][new_b] > current_sum + cost:
                            new_dp[new_a][new_b] = min(new_dp[new_a][new_b], current_sum + cost)
                    
                    # Option 3: Apply op2
                    if b_prev > 0 and x >= k:
                        cost = x - k
                        new_a, new_b = a_prev, b_prev - 1
                        if new_dp[new_a][new_b] > current_sum + cost:
                            new_dp[new_a][new_b] = min(new_dp[new_a][new_b], current_sum + cost)
                    
                    # Option 4: Apply both
                    if a_prev > 0 and b_prev > 0:
                        options = []
                        # Option 4a: op2 then op1
                        if x >= k:
                            val = x - k
                            cost1 = (val + 1) // 2
                            options.append(cost1)
                        # Option 4b: op1 then op2
                        x1 = (x + 1) // 2
                        if x1 >= k:
                            cost2 = x1 - k
                            options.append(cost2)
                        if options:
                            cost = min(options)
                            new_a, new_b = a_prev - 1, b_prev - 1
                            if new_dp[new_a][new_b] > current_sum + cost:
                                new_dp[new_a][new_b] = min(new_dp[new_a][new_b], current_sum + cost)
            
            dp = new_dp  # Update dp to new_dp for the next iteration
        
        # Find the minimal sum in the final dp table
        min_sum = INF
        for a in range(max_op1 + 1):
            for b in range(max_op2 + 1):
                if dp[a][b] < min_sum:
                    min_sum = dp[a][b]
        return min_sum