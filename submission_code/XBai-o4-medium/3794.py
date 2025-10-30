from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Initialize the previous row (i=0)
        prev = [0] * m
        prev[0] = skill[0] * mana[0]
        for j in range(1, m):
            prev[j] = prev[j-1] + skill[0] * mana[j]
        
        # Iterate over the remaining wizards
        for i in range(1, n):
            curr = [0] * m
            # Compute curr[0]
            curr[0] = prev[0] + skill[i] * mana[0]
            # Compute the rest
            for j in range(1, m):
                curr[j] = max(prev[j], curr[j-1]) + skill[i] * mana[j]
            prev = curr
        
        return prev[-1]