class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        INF = 10**18
        dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        for num in nums:
            new_dp = [[INF] * (op2 + 1) for _ in range(op1 + 1)]
            for a in range(op1 + 1):
                for b in range(op2 + 1):
                    if dp[a][b] == INF:
                        continue
                    current_sum = dp[a][b]
                    
                    # Choice 0: no operation
                    if current_sum + num < new_dp[a][b]:
                        new_dp[a][b] = current_sum + num
                    
                    # Choice 1: only op1
                    if a < op1:
                        val1 = (num + 1) // 2
                        total = current_sum + val1
                        if total < new_dp[a+1][b]:
                            new_dp[a+1][b] = total
                    
                    # Choice 2: only op2
                    if b < op2 and num >= k:
                        val2 = num - k
                        total = current_sum + val2
                        if total < new_dp[a][b+1]:
                            new_dp[a][b+1] = total
                    
                    # Choice 3: both op1 and op2
                    if a < op1 and b < op2:
                        candidates = []
                        if num >= k:
                            candidate1 = (num - k + 1) // 2
                            candidates.append(candidate1)
                        val1_first = (num + 1) // 2
                        if val1_first >= k:
                            candidate2 = val1_first - k
                            candidates.append(candidate2)
                        if candidates:
                            val3 = min(candidates)
                            total = current_sum + val3
                            if total < new_dp[a+1][b+1]:
                                new_dp[a+1][b+1] = total
            dp = new_dp
        
        ans = INF
        for a in range(op1 + 1):
            for b in range(op2 + 1):
                if dp[a][b] < ans:
                    ans = dp[a][b]
        return ans