import math
from typing import List
# import itertools # Optional: can use itertools.accumulate for prefix sums

class Solution:
  """
  Calculates the minimum time required for n wizards to brew m potions sequentially.

  Each potion j has a mana capacity mana[j] and must pass through wizards 0 to n-1.
  The time taken by wizard i on potion j is skill[i] * mana[j].
  Potions must be passed immediately from one wizard to the next. This implies synchronization:
  Wizard i+1 starts working on potion j exactly when wizard i finishes potion j.
  Also, each wizard i cannot start working on potion j until they have finished working on potion j-1.
  The goal is to find the minimum time when the last wizard (n-1) finishes the last potion (m-1).
  """
  def minTime(self, skill: List[int], mana: List[int]) -> int:
    """
    Args:
      skill: A list of integers representing the skill level of each wizard. Length n.
      mana: A list of integers representing the mana capacity of each potion. Length m.

    Returns:
      The minimum total time required to brew all potions.
    """
    n = len(skill)
    m = len(mana)

    # Constraints state n, m >= 1, so no need to handle empty lists explicitly.
    
    # Precompute prefix sums of skill levels. 
    # prefix_skill[i] = sum(skill[0]...skill[i])
    # This represents the total time taken by wizards 0 to i for a potion with mana capacity 1.
    prefix_skill = [0] * n
    current_sum = 0
    for i in range(n):
        current_sum += skill[i]
        prefix_skill[i] = current_sum
    
    # T array stores the finish times for each wizard on the *current* potion being processed.
    # T[i] will hold the finish time T_{i, j} after processing potion j.
    # We only need to keep track of the finish times for the most recently completed potion
    # to calculate the start time for the next one. So, one array of size n is sufficient.
    T = [0] * n

    # Compute finish times for the first potion (j=0).
    # The first potion starts processing at time 0. S_{0,0} = 0.
    # The finish time for wizard i on potion 0 is T_{i,0} = S_{0,0} + mana[0] * prefix_skill[i]
    # Since S_{0,0} = 0, T_{i,0} = mana[0] * prefix_skill[i].
    for i in range(n):
        # Python integers handle arbitrary size, preventing overflow.
        T[i] = mana[0] * prefix_skill[i]

    # Iterate through the remaining potions from j = 1 to m-1
    for j in range(1, m):
        # Calculate the earliest possible start time S_{0,j} for wizard 0 on the current potion j.
        # This start time is constrained by the finish times T_{i, j-1} of all wizards on the previous potion (j-1).
        # The values T_{i, j-1} are currently stored in the T array from the previous iteration.
        
        # The constraints are:
        # 1. Wizard 0 must be free: S_{0,j} >= T_{0, j-1}
        # 2. Wizard i must be free when potion j arrives: S_{i,j} >= T_{i, j-1}
        #    Since S_{i,j} = T_{i-1, j} = S_{0,j} + mana[j] * prefix_skill[i-1], this becomes:
        #    S_{0,j} + mana[j] * prefix_skill[i-1] >= T_{i, j-1}
        #    Which implies S_{0,j} >= T_{i, j-1} - mana[j] * prefix_skill[i-1] for i = 1..n-1
        
        # To minimize total time, we choose the minimum S_{0,j} satisfying all constraints.
        # S_{0,j} = max( T_{0, j-1}, max_{1 <= i <= n-1} (T_{i, j-1} - mana[j] * prefix_skill[i-1}) )
        
        # Initialize S0j with the first constraint based on wizard 0. T[0] holds T_{0, j-1}.
        S0j = T[0] 
        
        # Apply constraints for wizards i = 1 to n-1.
        for i in range(1, n):
            # T[i] holds T_{i, j-1}. prefix_skill[i-1] holds P_{i-1} = sum(skill[0]..skill[i-1]).
            required_start_time = T[i] - mana[j] * prefix_skill[i-1]
            # Update S0j to the maximum required start time found so far.
            S0j = max(S0j, required_start_time)

        # Now we have the minimum valid start time S_{0,j} for potion j.
        # Compute the finish times T_{i,j} for the current potion j.
        # The finish time for wizard i is T_{i,j} = S_{0,j} + mana[j] * prefix_skill[i]
        # Update the T array in place to store these new finish times T_{i,j}.
        # These values will then be used as T_{i, j} in the next iteration (for potion j+1).
        for i in range(n):
            T[i] = S0j + mana[j] * prefix_skill[i]

    # After processing all potions up to m-1, T[n-1] contains the finish time
    # of the last wizard (n-1) on the last potion (m-1). This is the minimum total time required.
    return T[n-1]