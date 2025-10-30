class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of healthy children between each pair of sick children
        gaps = [sick[0]] + [sick[i] - sick[i-1] - 1 for i in range(1, len(sick))] + [n - sick[-1] - 1]
        
        # Total number of healthy children
        k = n - len(sick)
        
        # Initialize result with binomial coefficient (k choose k)
        result = 1
        
        # Calculate 2^(sum of internal gaps)
        power_of_2 = pow(2, sum(gaps[1:-1]), MOD)
        result = (result * power_of_2) % MOD
        
        # Calculate contribution of each gap
        for i, gap in enumerate(gaps):
            if gap > 0:
                # If it's an internal gap, divide by 2^(gap-1)
                if 0 < i < len(gaps) - 1:
                    result = (result * pow(2, MOD - 1 - (gap - 1), MOD)) % MOD
                
                # Calculate binomial coefficient (k choose gap)
                for j in range(gap):
                    result = (result * (k - j)) % MOD
                    result = (result * pow(j + 1, MOD - 2, MOD)) % MOD
                
                k -= gap
        
        return result