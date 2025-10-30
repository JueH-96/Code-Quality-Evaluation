from typing import List
from math import comb

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        m = len(sick)
        total_sequences = 1
        
        # Calculate the number of gaps between sick children and at the ends
        gaps = [sick[0], n - 1 - sick[-1]]
        for i in range(1, m):
            gaps.append(sick[i] - sick[i - 1] - 1)
        
        # Calculate the number of ways to arrange the non-infected children
        total_non_infected = n - m
        current_non_infected = 0
        
        for gap in gaps:
            if gap > 0:
                total_sequences = (total_sequences * comb(total_non_infected, gap)) % MOD
                total_non_infected -= gap
        
        return total_sequences