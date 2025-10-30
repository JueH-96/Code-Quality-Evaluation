from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        # Compute prefix sums of skill
        sp = [0] * n
        sp[0] = skill[0]
        for i in range(1, n):
            sp[i] = sp[i - 1] + skill[i]
        
        # start0 will track the start time of each potion on wizard 0
        start0 = 0
        # For each potion j>0, compute the minimal gap D_j needed to satisfy
        # the no‐wait constraints across all wizards
        for j in range(1, m):
            xprev = mana[j - 1]
            xcurr = mana[j]
            best = 0
            prev_sp = 0
            # B_i = sum_{k=0..i} p[k][j-1] - sum_{k=0..i-1} p[k][j]
            #      = sp[i]*xprev - prev_sp*xcurr
            for sp_i in sp:
                B = sp_i * xprev - prev_sp * xcurr
                if B > best:
                    best = B
                prev_sp = sp_i
            start0 += best
        
        # Finally add the total processing time of the last potion on all wizards
        return start0 + sp[-1] * mana[-1]