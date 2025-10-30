from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)
        
        # Compute cumulative sum for skills
        cum_sum = [0]
        current_sum = 0
        for s in skill:
            current_sum += s
            cum_sum.append(current_sum)
        
        # Total skill sum
        total_skill = cum_sum[n]
        
        # Compute start times for each potion on wizard 0
        t_start = [0] * m
        if m > 1:
            for idx in range(1, m):
                mana_prev = mana[idx - 1]
                mana_curr = mana[idx]
                delta_mana = mana_prev - mana_curr
                # Compute the minimum difference using max over k
                min_diff = max(skill[k] * mana_prev + cum_sum[k] * delta_mana for k in range(n))
                t_start[idx] = t_start[idx - 1] + min_diff
        
        # Compute the maximum completion time
        return max(t_start[j] + total_skill * mana[j] for j in range(m))