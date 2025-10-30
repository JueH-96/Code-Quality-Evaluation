class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        INF = 10**18
        dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for num in nums:
            choices = []
            choices.append((0, 0, num))
            choices.append((1, 0, (num + 1) // 2))
            if k > 0 and num >= k:
                choices.append((0, 1, num - k))
            if k > 0 and num >= k:
                op1_val = (num + 1) // 2
                op2_then_op1 = (num - k + 1) // 2
                op1_then_op2_val = op1_val - k if op1_val >= k else INF
                both_val = min(op1_then_op2_val, op2_then_op1)
                choices.append((1, 1, both_val))
            
            new_dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
            for a in range(op1 + 1):
                for b in range(op2 + 1):
                    if dp[a][b] == INF:
                        continue
                    for (u1, u2, val) in choices:
                        na, nb = a + u1, b + u2
                        if na <= op1 and nb <= op2:
                            if new_dp[na][nb] > dp[a][b] + val:
                                new_dp[na][nb] = dp[a][b] + val
            dp = new_dp
        
        ans = INF
        for a in range(op1 + 1):
            for b in range(op2 + 1):
                if dp[a][b] < ans:
                    ans = dp[a][b]
        return ans