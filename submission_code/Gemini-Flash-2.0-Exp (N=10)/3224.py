import math

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        healthy = []
        sick_set = set(sick)
        for i in range(n):
            if i not in sick_set:
                healthy.append(i)
        
        if not healthy:
            return 0
        
        gaps = []
        
        if healthy[0] != 0:
            gaps.append(healthy[0])
        
        for i in range(len(healthy) - 1):
            gaps.append(healthy[i+1] - healthy[i] - 1)
        
        if healthy[-1] != n - 1:
            gaps.append(n - 1 - healthy[-1])
        
        
        total_healthy = len(healthy)
        
        ans = 1
        
        for gap in gaps:
            if gap > 0:
                ans = (ans * pow(2, gap - 1, MOD)) % MOD
        
        
        
        def factorial(num):
            res = 1
            for i in range(1, num + 1):
                res = (res * i) % MOD
            return res
        
        
        ans = (ans * factorial(total_healthy)) % MOD
        
        return ans