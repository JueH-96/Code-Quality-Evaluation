class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Find the gaps between sick children
        gaps = []
        for i in range(1, len(sick)):
            gaps.append(sick[i] - sick[i-1] - 1)
        
        # Add the gaps at the beginning and end
        if sick[0] > 0:
            gaps.insert(0, sick[0])
        if sick[-1] < n - 1:
            gaps.append(n - 1 - sick[-1])
        
        # Calculate the factorial and modular inverse factorial
        def factorial(n):
            result = 1
            for i in range(2, n + 1):
                result = (result * i) % MOD
            return result
        
        def mod_inverse(x):
            return pow(x, MOD - 2, MOD)
        
        total_gaps = sum(gaps)
        fact_total_gaps = factorial(total_gaps)
        
        result = fact_total_gaps
        for gap in gaps:
            result = (result * mod_inverse(factorial(gap))) % MOD
        
        return result