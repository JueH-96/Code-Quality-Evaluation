class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        low, high = 1, n
        ans = high
        while low <= high:
            mid = (low + high) // 2
            if self.feasible(s, numOps, mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

    def feasible(self, s, numOps, x):
        n = len(s)
        INF = 10**9
        dp0 = [(INF, 0)] * n
        dp1 = [(INF, 0)] * n
        
        if s[0] == '0':
            dp0[0] = (0, 1)
            dp1[0] = (1, 1)
        else:
            dp0[0] = (1, 1)
            dp1[0] = (0, 1)
        
        for i in range(1, n):
            candidates0 = []
            if dp0[i-1][1] < x:
                new_run = dp0[i-1][1] + 1
                cost = dp0[i-1][0] + (0 if s[i] == '0' else 1)
                candidates0.append((cost, new_run))
            cost_break = dp1[i-1][0] + (0 if s[i] == '0' else 1)
            candidates0.append((cost_break, 1))
            candidates0.sort(key=lambda cand: (cand[0], cand[1]))
            dp0[i] = candidates0[0]
            
            candidates1 = []
            if dp1[i-1][1] < x:
                new_run = dp1[i-1][1] + 1
                cost = dp1[i-1][0] + (0 if s[i] == '1' else 1)
                candidates1.append((cost, new_run))
            cost_break = dp0[i-1][0] + (0 if s[i] == '1' else 1)
            candidates1.append((cost_break, 1))
            candidates1.sort(key=lambda cand: (cand[0], cand[1]))
            dp1[i] = candidates1[0]
            
        min_ops = min(dp0[-1][0], dp1[-1][0])
        return min_ops <= numOps