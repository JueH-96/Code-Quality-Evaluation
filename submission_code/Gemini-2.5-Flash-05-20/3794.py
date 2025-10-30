import math
from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)

        # 1. Precompute prefix sums of skill levels.
        # prefix_skill_sum[i] = skill[0] + ... + skill[i]
        prefix_skill_sum = [0] * n
        prefix_skill_sum[0] = skill[0]
        for i in range(1, n):
            prefix_skill_sum[i] = prefix_skill_sum[i-1] + skill[i]

        # 2. Initialize last_potion_finish_times.
        # last_potion_finish_times[i] stores F[i][j-1], the time wizard 'i' finished the previous potion.
        # Initially, assume all wizards finished a "virtual" potion at time 0.
        last_potion_finish_times = [0] * n

        # Loop through each potion (j) from 0 to m-1.
        for j in range(m):
            current_mana = mana[j]
            
            # 3a. Determine current_potion_start_time (S_j).
            # This is the global start time for the current potion.
            current_potion_start_time = 0 
            
            if j == 0:
                # The first potion starts at time 0.
                current_potion_start_time = 0
            else:
                # For j > 0, S_j must satisfy:
                # S_j + (sum_{p=0 to k-1} skill[p]) * mana[j] >= F[k][j-1] for all k in [0, n-1]
                # This ensures wizard k is free when potion j arrives at them from wizard k-1 (or global start for k=0).
                # Rearranging: S_j >= F[k][j-1] - (sum_{p=0 to k-1} skill[p]) * mana[j]
                # S_j must be the maximum of these required start times, and also at least 0.
                
                # Iterate k from 0 to n-1 to find the maximum required S_j.
                for k in range(n):
                    # time_to_reach_wizard_k:
                    # This is the accumulated time taken by wizards 0 to k-1 on potion j,
                    # assuming potion j started at S_j and passed immediately.
                    # For k=0, this term is 0. For k>0, it's prefix_skill_sum[k-1] * current_mana.
                    time_to_reach_wizard_k = (prefix_skill_sum[k-1] if k > 0 else 0) * current_mana
                    
                    # Calculate the minimum S_j required for wizard k
                    required_s_j_for_k = last_potion_finish_times[k] - time_to_reach_wizard_k
                    
                    # Update the overall current_potion_start_time (S_j)
                    current_potion_start_time = max(current_potion_start_time, required_s_j_for_k)

            # 3b. Calculate current_potion_finish_times (F[i][j]) for the current potion.
            current_potion_finish_times = [0] * n
            for i in range(n):
                # F[i][j] = S_j + (skill[0] + ... + skill[i]) * mana[j]
                time_spent_by_wizards_up_to_i = prefix_skill_sum[i] * current_mana
                current_potion_finish_times[i] = current_potion_start_time + time_spent_by_wizards_up_to_i
            
            # 3c. Update last_potion_finish_times for the next iteration (next potion).
            last_potion_finish_times = current_potion_finish_times

        # 4. The final answer is the time the last wizard (n-1) finishes the last potion (m-1).
        return last_potion_finish_times[n-1]