from typing import List
import numpy as np

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        # number of wizards and potions
        n = len(skill)
        m = len(mana)
        
        # Convert to numpy arrays (using int64 to be safe).
        skill_arr = np.array(skill, dtype=np.int64)
        mana_arr = np.array(mana, dtype=np.int64)
        
        # Compute the prefix sums: P[i] = skill[0] + ... + skill[i]
        prefix = np.cumsum(skill_arr)  # shape (n,)
        
        # For convenience, let P_shift be an array where
        # P_shift[0] = 0 and for i>=1, P_shift[i] = prefix[i-1].
        P_shift = np.empty_like(prefix)
        P_shift[0] = 0
        if n > 1:
            P_shift[1:] = prefix[:-1]
        
        total_delay = 0  # This will sum the delays between potions
        
        # For each transition from potion j to potion j+1, we must add a delay D[j].
        # The condition we derived says:
        #    D[j] = max_{0 <= r < n} { (prefix[r]) * mana[j] - (P_shift[r]) * mana[j+1] }
        for j in range(m - 1):
            # For current potion j and next potion j+1,
            # compute the candidate delay for each r in [0, n-1].
            # (This is vectorized: it creates an array of length n with the value for each r.)
            candidate = prefix * mana_arr[j] - P_shift * mana_arr[j+1]
            # The required extra delay is the maximum value
            delay = int(candidate.max())
            total_delay += delay
        
        # The overall finish time is the start time of the last potion (which is the sum
        # of all delays) plus the time to process the last potion on all wizards.
        answer = total_delay + int(prefix[-1] * mana_arr[m - 1])
        return answer