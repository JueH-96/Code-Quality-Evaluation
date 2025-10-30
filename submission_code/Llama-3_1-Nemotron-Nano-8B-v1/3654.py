from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        def compute_both(x: int) -> int:
            case_a_val = None
            if x >= k:
                after_op2 = x - k
                after_op1 = (after_op2 + 1) // 2
                case_a_val = after_op1
            a = (x + 1) // 2
            case_b_val = a - k if a >= k else None
            possible = []
            if case_a_val is not None:
                possible.append(case_a_val)
            if case_b_val is not None:
                possible.append(case_b_val)
            return min(possible) if possible else None
        
        elements_options = []
        for x in nums:
            options = []
            # Do nothing
            options.append((x, 0, 0))
            # Do op1
            a = (x + 1) // 2
            options.append((a, 1, 0))
            # Do op2 if possible
            if x >= k:
                options.append((x - k, 0, 1))
            # Do both if possible
            both_val = compute_both(x)
            if both_val is not None:
                options.append((both_val, 1, 1))
            elements_options.append(options)
        
        INF = float('inf')
        dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0  # Initial state: sum is 0, no operations applied
        
        for options in elements_options:
            temp_dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
            for i in range(op1 + 1):
                for j in range(op2 + 1):
                    if dp[i][j] == INF:
                        continue
                    for cost, o1, o2 in options:
                        new_i = i + o1
                        new_j = j + o2
                        if new_i > op1 or new_j > op2:
                            continue
                        if dp[i][j] + cost < temp_dp[new_i][new_j]:
                            temp_dp[new_i][new_j] = dp[i][j] + cost
            dp = temp_dp
        
        min_sum = INF
        for i in range(op1 + 1):
            for j in range(op2 + 1):
                if dp[i][j] < min_sum:
                    min_sum = dp[i][j]
        return min_sum if min_sum != INF else 0