from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        # S_prev is the start time of the j-th potion; S_0 = 0
        S_prev = 0
        
        # For each potion j from 1 to m-1, compute the minimal delay S_j - S_{j-1}
        # needed so that no-wait and no-machine-conflict constraints hold.
        for j in range(1, m):
            prev_pref = 0    # prefix sum of p[i][j-1]
            cur_ex = 0       # prefix sum of p[k][j] for k < i
            max_cand = 0     # we take max over i of (prev_pref - cur_ex)
            
            # Iterate over wizards i = 0..n-1
            # p[i][j-1] = skill[i] * mana[j-1]
            # p[i][j]   = skill[i] * mana[j]
            m_prev = mana[j-1]
            m_cur = mana[j]
            for i in range(n):
                prev_pref += skill[i] * m_prev
                # candidate delay to keep machine i free when potion j arrives
                cand = prev_pref - cur_ex
                if cand > max_cand:
                    max_cand = cand
                cur_ex += skill[i] * m_cur
            
            # the start time of potion j
            S_prev += max_cand
        
        # After determining S_{m-1}, we add the total processing time of the last potion
        # across all wizards to get the makespan.
        total_last = 0
        last_m = mana[m-1]
        for i in range(n):
            total_last += skill[i] * last_m
        
        return S_prev + total_last