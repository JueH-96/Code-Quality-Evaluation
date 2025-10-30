class Solution:
    def makeStringGood(self, s: str) -> int:
        n = len(s)
        a = [0] * 26
        for ch in s:
            a[ord(ch) - ord('a')] += 1
        
        ans = float('inf')
        for f in range(0, n + 1):
            dp = [float('inf')] * (n + 1)
            dp[0] = 0
            for i in range(25):
                new_dp = [float('inf')] * (n + 1)
                for inflow, cost_val in enumerate(dp):
                    if cost_val == float('inf'):
                        continue
                    current = a[i] + inflow
                    cost0 = current
                    next_in0 = current
                    if next_in0 <= n:
                        if cost_val + cost0 < new_dp[next_in0]:
                            new_dp[next_in0] = cost_val + cost0
                    if f > 0:
                        if current >= f:
                            costf = current - f
                            next_inf = current - f
                        else:
                            costf = f - current
                            next_inf = 0
                        if cost_val + costf < new_dp[next_inf]:
                            new_dp[next_inf] = cost_val + costf
                dp = new_dp
            for inflow, cost_val in enumerate(dp):
                if cost_val == float('inf'):
                    continue
                current_last = a[25] + inflow
                cost_last = min(current_last, abs(current_last - f))
                total_cost = cost_val + cost_last
                if total_cost < ans:
                    ans = total_cost
        return ans