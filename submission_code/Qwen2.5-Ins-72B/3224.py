from math import comb
from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Calculate the number of ways to infect the children
        def count_ways(gaps):
            total_ways = 1
            total_gaps = sum(gaps)
            for gap in gaps:
                total_ways = (total_ways * comb(total_gaps, gap)) % MOD
                total_gaps -= gap
            return total_ways
        
        # Calculate the gaps between sick children
        gaps = []
        for i in range(len(sick) - 1):
            if sick[i + 1] - sick[i] > 1:
                gaps.append(sick[i + 1] - sick[i] - 1)
        
        # Calculate the number of ways to infect the children
        total_ways = count_ways(gaps)
        
        # Calculate the number of ways to infect the children at the ends
        if sick[0] > 0:
            total_ways = (total_ways * comb(sick[0] + len(gaps), sick[0])) % MOD
        if sick[-1] < n - 1:
            total_ways = (total_ways * comb(n - 1 - sick[-1] + len(gaps), n - 1 - sick[-1])) % MOD
        
        return total_ways