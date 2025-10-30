from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Initialize the first row (wizard 0)
        dp_prev = [0] * m
        dp_prev[0] = skill[0] * mana[0]
        for j in range(1, m):
            dp_prev[j] = dp_prev[j-1] + skill[0] * mana[j]
        
        for i in range(1, n):
            dp_current = [0] * m
            # Process each potion j for the current wizard i
            dp_current[0] = dp_prev[0] + skill[i] * mana[0]
            for j in range(1, m):
                dp_current[j] = max(dp_current[j-1], dp_prev[j]) + skill[i] * mana[j]
            dp_prev = dp_current
        
        return max(dp_prev)