from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        result = 1
        gaps = []
        prev_sick = -1
        
        for i in sick:
            if i - prev_sick > 1:
                gaps.append(i - prev_sick - 1)
            prev_sick = i
        
        if n - prev_sick - 1 > 0:
            gaps.append(n - prev_sick - 1)
        
        for gap in gaps:
            if gap > 1:
                result = (result * pow(2, gap - 1, MOD) * self.binomial_coefficient(gap + gap - 1, gap - 1, MOD)) % MOD
        
        return result
    
    def binomial_coefficient(self, n: int, k: int, MOD: int) -> int:
        if k > n - k:
            k = n - k
        result = 1
        for i in range(k):
            result = result * (n - i) % MOD
            result = result * pow(i + 1, MOD - 2, MOD) % MOD
        return result