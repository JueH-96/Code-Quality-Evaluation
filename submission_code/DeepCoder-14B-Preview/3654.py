class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        def get_options(x, k_val):
            options = []
            # Option a: no operations
            options.append((x, 0, 0))
            # Option b: only op1
            new_val = (x + 1) // 2
            options.append((new_val, 1, 0))
            # Option c: only op2 if x >= k
            if x >= k_val:
                new_val = x - k_val
                options.append((new_val, 0, 1))
            # Option d: op1 then op2 if after op1 the value >= k
            new_val_after_op1 = (x + 1) // 2
            if new_val_after_op1 >= k_val:
                new_val = new_val_after_op1 - k_val
                options.append((new_val, 1, 1))
            # Option e: op2 then op1 if x >= k
            if x >= k_val:
                new_val_after_op2 = x - k_val
                new_val_after_op1 = (new_val_after_op2 + 1) // 2
                options.append((new_val_after_op1, 1, 1))
            return options
        
        INF = float('inf')
        dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0  # Initial state
        
        for x in nums:
            new_dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
            for i in range(op1 + 1):
                for j in range(op2 + 1):
                    if dp[i][j] == INF:
                        continue
                    for val, a, b in get_options(x, k):
                        ni = i + a
                        nj = j + b
                        if ni > op1 or nj > op2:
                            continue
                        if new_dp[ni][nj] > dp[i][j] + val:
                            new_dp[ni][nj] = dp[i][j] + val
            dp = new_dp
        
        min_sum = INF
        for i in range(op1 + 1):
            for j in range(op2 + 1):
                if dp[i][j] < min_sum:
                    min_sum = dp[i][j]
        return min_sum if min_sum != INF else 0