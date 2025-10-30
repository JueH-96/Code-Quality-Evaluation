from typing import List

class Solution:
  def minTime(self, skill: List[int], mana: List[int]) -> int:
    n = len(skill)
    m = len(mana)

    # Constraints: 1 <= n, m <= 5000. So n and m are at least 1.
    # No need to check for n=0 or m=0 explicitly.

    # Calculate prefix sums of skill
    # prefix_skill_sum[i] stores skill[0] + ... + skill[i]
    prefix_skill_sum = [0] * n
    prefix_skill_sum[0] = skill[0]
    for i in range(1, n):
        prefix_skill_sum[i] = prefix_skill_sum[i-1] + skill[i]
    
    # dp[k_wizard] stores finish_time[k_wizard][j_potion-1]
    # (i.e., finish time for wizard k_wizard on the previously processed potion)
    # Initialize dp for a hypothetical potion indexed -1 that finishes at time 0 for all wizards.
    dp = [0] * n 

    # Iterate through potions
    for j_potion in range(m): # current potion index, from 0 to m-1
        current_mana = mana[j_potion]
        
        current_S_j: int # This will be the start time for wizard 0 on potion j_potion
        
        if j_potion == 0:
            # For the first potion (j_potion = 0), wizard 0 can start at time 0.
            # There is no potion j_potion-1, so constraints related to previous potion's finish times
            # effectively mean S_0 >= 0. The minimum non-negative S_0 is 0.
            current_S_j = 0
        else:
            # Calculate S_j for j_potion > 0.
            # S_j must be large enough to satisfy all constraints.
            # dp[k_wizard] currently holds finish_time[k_wizard][j_potion-1].

            # Constraint 1a: Wizard 0 must be free.
            # start_time[0][j_potion] >= finish_time[0][j_potion-1]
            # current_S_j >= dp[0]
            current_S_j = dp[0] 
            
            # Constraint 1b: Wizard i_wizard (for i_wizard > 0) must be free.
            # start_time[i_wizard][j_potion] >= finish_time[i_wizard][j_potion-1]
            # current_S_j + prefix_skill_sum[i_wizard-1]*current_mana >= dp[i_wizard]
            #   =>  current_S_j >= dp[i_wizard] - prefix_skill_sum[i_wizard-1]*current_mana
            # This loop covers wizard indices i_wizard from 1 to n-1.
            for i_wizard in range(1, n):
                # prefix_skill_sum[i_wizard-1] is sum of skills for wizards 0 to i_wizard-1
                val = dp[i_wizard] - prefix_skill_sum[i_wizard-1] * current_mana
                if val > current_S_j:
                    current_S_j = val
            
            # Constraint 2: Wizard i_wizard+1 must be free when wizard i_wizard passes potion j_potion.
            # finish_time[i_wizard][j_potion] >= finish_time[i_wizard+1][j_potion-1]
            # current_S_j + prefix_skill_sum[i_wizard]*current_mana >= dp[i_wizard+1]
            #   =>  current_S_j >= dp[i_wizard+1] - prefix_skill_sum[i_wizard]*current_mana
            # This loop covers wizard indices i_wizard from 0 to n-2.
            # (So i_wizard+1 goes from 1 to n-1)
            for i_wizard in range(n - 1): 
                # prefix_skill_sum[i_wizard] is sum of skills for wizards 0 to i_wizard
                val = dp[i_wizard+1] - prefix_skill_sum[i_wizard] * current_mana
                if val > current_S_j:
                    current_S_j = val
        
        # At this point, current_S_j is the minimum valid start time for wizard 0 on potion j_potion.
        # It is guaranteed to be non-negative because dp[0] (if j_potion > 0) is a finish time and thus non-negative.
        # If any (dp[k] - C*mana) term is negative, it's a non-binding constraint if S_j >= 0.

        # Calculate finish_time[k_wizard][j_potion] for all wizards k_wizard.
        # Store these new finish times in dp to be used for the next potion (j_potion+1).
        # finish_time[k_wizard][j_potion] = current_S_j + prefix_skill_sum[k_wizard]*current_mana
        for k_wizard in range(n): # k_wizard from 0 to n-1
            dp[k_wizard] = current_S_j + prefix_skill_sum[k_wizard] * current_mana
            
    # The result is the finish time of the last wizard (n-1) on the last potion (m-1).
    # This is stored in dp[n-1] after all potions are processed.
    return dp[n-1]