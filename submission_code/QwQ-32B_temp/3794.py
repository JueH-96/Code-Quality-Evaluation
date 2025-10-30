from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        if n == 0 or m == 0:
            return 0
        
        # Initialize the first row (i=0)
        prev_S = [0] * m
        for j in range(1, m):
            prev_S[j] = prev_S[j-1] + skill[0] * mana[j-1]
        
        max_time = 0
        # Compute end times for the first row
        for j in range(m):
            end = prev_S[j] + skill[0] * mana[j]
            if end > max_time:
                max_time = end
        
        # Process each subsequent wizard
        for i in range(1, n):
            current_S = [0] * m
            # Compute current_S[0]
            current_S[0] = prev_S[0] + skill[i-1] * mana[0]
            end0 = current_S[0] + skill[i] * mana[0]
            if end0 > max_time:
                max_time = end0
            
            # Compute the rest of current_S[j] for j from 1 to m-1
            for j in range(1, m):
                term1 = prev_S[j] + skill[i-1] * mana[j]
                term2 = current_S[j-1] + skill[i] * mana[j-1]
                current_S[j] = max(term1, term2)
                end_j = current_S[j] + skill[i] * mana[j]
                if end_j > max_time:
                    max_time = end_j
            
            # Update prev_S to current_S for next iteration
            prev_S = current_S.copy()
        
        return max_time