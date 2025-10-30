import itertools
from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)

        # prefix_skill[i] = skill[0] + ... + skill[i]
        prefix_skill = list(itertools.accumulate(skill))

        # S_j is the start time for wizard 0 on potion j.
        # We want to find S_{m-1}.
        # S_0 is 0, so we start with current_start_time = 0.
        current_start_time = 0

        # Loop through potions from the second one (j=1) to calculate their start times.
        for j in range(1, m):
            mana_prev = mana[j-1]
            mana_curr = mana[j]
            
            # The time difference between the start of potion j and j-1 is delta_S_j.
            # To satisfy synchronization for all wizards, delta_S_j must be at least:
            # max_{i=0..n-1} ( prefix_skill[i]*mana_prev - prefix_skill[i-1]*mana_curr )
            # where prefix_skill[-1] is defined as 0.

            # Initialize max_val with the term for i = 0.
            # The term is: prefix_skill[0]*mana_prev - prefix_skill[-1]*mana_curr
            # Since prefix_skill[-1] = 0, this simplifies to prefix_skill[0]*mana_prev.
            max_val = prefix_skill[0] * mana_prev
            
            # Loop for i = 1 to n-1 to find the maximum value.
            for i in range(1, n):
                val = prefix_skill[i] * mana_prev - prefix_skill[i-1] * mana_curr
                if val > max_val:
                    max_val = val
            
            delta_S_j = max_val
            current_start_time += delta_S_j

        # The total time is the finish time of the last wizard on the last potion.
        # finish_time(n-1, m-1) = S_{m-1} + prefix_skill[n-1] * mana[m-1]
        # Here, current_start_time represents S_{m-1}.
        
        total_time = current_start_time + prefix_skill[n-1] * mana[m-1]
        
        return total_time