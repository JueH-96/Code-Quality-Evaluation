class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        s_int = [1 if ch == '1' else 0 for ch in s]
        
        def can_achieve(x):
            INF = 10**9
            dp = [[INF] * (x + 1) for _ in range(2)]
            first_char = s_int[0]
            dp[first_char][1] = 0
            dp[1 - first_char][1] = 1
            
            for i in range(1, n):
                next_dp = [[INF] * (x + 1) for _ in range(2)]
                current_char = s_int[i]
                for last in range(2):
                    for run in range(1, x + 1):
                        current_ops = dp[last][run]
                        if current_ops == INF:
                            continue
                        if current_char == last:
                            if run + 1 <= x:
                                if current_ops < next_dp[last][run + 1]:
                                    next_dp[last][run + 1] = current_ops
                            new_ops = current_ops + 1
                            if new_ops < next_dp[1 - last][1]:
                                next_dp[1 - last][1] = new_ops
                        else:
                            if current_ops < next_dp[current_char][1]:
                                next_dp[current_char][1] = current_ops
                            if run + 1 <= x:
                                new_ops = current_ops + 1
                                if new_ops < next_dp[last][run + 1]:
                                    next_dp[last][run + 1] = new_ops
                dp = next_dp
            
            for last in range(2):
                for run in range(1, x + 1):
                    if dp[last][run] <= numOps:
                        return True
            return False
        
        low, high = 1, n
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if can_achieve(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans