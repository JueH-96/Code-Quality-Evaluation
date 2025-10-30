from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        def factorial(x):
            result = 1
            for i in range(1, x + 1):
                result = (result * i) % MOD
            return result
        
        def count_sequences(gap):
            # Number of ways to arrange infections in a gap of size 'gap'
            return factorial(gap)
        
        total_ways = 1
        prev_sick = -1
        for i in range(len(sick)):
            if prev_sick != -1:
                gap = sick[i] - prev_sick - 1
                total_ways = (total_ways * count_sequences(gap)) % MOD
            prev_sick = sick[i]
        
        # Handle the gap at the end if it exists
        if sick[-1] != n - 1:
            gap = n - 1 - sick[-1]
            total_ways = (total_ways * count_sequences(gap)) % MOD
        
        return total_ways

# Example usage:
# sol = Solution()
# print(sol.numberOfSequence(5, [0, 4]))  # Output: 4
# print(sol.numberOfSequence(4, [1]))    # Output: 3