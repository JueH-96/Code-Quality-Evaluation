import math
from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        """
        Calculates the minimum number of moves Alice needs to pick up exactly k ones.

        Args:
            nums: A binary array.
            k: The target number of ones to pick up.
            maxChanges: The maximum number of Type 1 moves (0 -> 1 changes) allowed.

        Returns:
            The minimum number of moves required.
        """

        # Find indices of all ones in the array
        pos = [i for i, x in enumerate(nums) if x == 1]
        M = len(pos)  # Number of existing ones
        n = len(nums) # Length of the array

        # Handle edge case: k = 0 (Technically k >= 1 based on constraints, but good practice)
        if k == 0:
            return 0

        # Handle edge case: No existing ones
        if M == 0:
            # Need k ones, must use Type 1 moves. Each costs 2 moves.
            # The constraint maxChanges + sum(nums) >= k guarantees maxChanges >= k here.
            return k * 2

        # Optimization: Check for trivial low-cost solutions for small k based on local patterns.
        # This section checks if k ones can be collected very cheaply due to clustering.
        min_moves_optimization = float('inf')

        # If k=1: Alice can start at any existing '1' and pick it up for free.
        if k == 1:
             # M > 0 is guaranteed here.
            min_moves_optimization = 0

        # Check k=2: Cost is 1 move if adjacent 1s (1,1) exist. Alice starts at one, moves the other adjacent one (dist 1).
        if k >= 2:
            has_adjacent = any(pos[i+1] - pos[i] == 1 for i in range(M - 1))
            if has_adjacent:
                 min_moves_optimization = min(min_moves_optimization, 1) # Needs 1 move for k=2.
        
        # Check k=2: Cost is 2 moves if pattern 1,0,1 exists (distance 2 between 1s).
        # Alice could start at the middle 0. Need 2 ones. Moves needed = dist 1 + dist 1 = 2.
        # Alternatively, Alice starts at one of the 1s, picks free, needs 1 more. The other 1 is dist 2 away. Cost 2 moves.
        if k >= 2: # This applies for k=2.
            has_one_gap = any(pos[i+1] - pos[i] == 2 for i in range(M - 1))
            if has_one_gap:
                min_moves_optimization = min(min_moves_optimization, 2)
        
        # Check k=3: Cost is 2 moves if pattern 1,1,1 exists.
        # Alice starts at the middle 1 (free pickup). Needs 2 more. Uses neighbors (dist 1 each). Total cost 1+1=2.
        if k >= 3:
            has_triple = any(pos[i+1] - pos[i] == 1 and pos[i+2] - pos[i+1] == 1 for i in range(M - 2))
            if has_triple:
                min_moves_optimization = min(min_moves_optimization, 2) # Needs 2 moves for k=3.

        # Compare optimization results with the baseline cost using only Type 1 changes.
        # If Alice starts at pos[i], picks free, needs k-1. Min cost using changes = 2*(k-1).
        # If Alice could potentially start at an index i where nums[i]=0, needs k. Min cost using changes = 2*k.
        # Note: The problem doesn't restrict Alice's starting position based on nums value. She can start anywhere.
        min_cost_using_only_changes = float('inf')
        if M > 0 and maxChanges >= k - 1: # If enough changes and can start at a '1'
             min_cost_using_only_changes = min(min_cost_using_only_changes, 2 * (k - 1))
        if maxChanges >= k: # If enough changes regardless of start point
             min_cost_using_only_changes = min(min_cost_using_only_changes, 2 * k)
        
        # Initialize overall minimum moves found so far
        min_total_moves = min(min_moves_optimization, min_cost_using_only_changes)

        # Main logic: Iterate through each existing '1' at index p=pos[i] as a potential aliceIndex.
        # This assumes the optimal aliceIndex is where an existing '1' is located.
        # This seems a reasonable heuristic, as starting at '1' grants a free pickup.
        for i in range(M):
            center_p = pos[i] # Alice starts at this index p = pos[i]
            
            # Alice picks up the '1' at center_p for free.
            # Needs target_k = k-1 more ones.
            target_k = k - 1
            if target_k == 0: # If k=1 originally
                 min_total_moves = 0
                 # Optimal cost 0 found, cannot do better.
                 return 0 # Early exit for k=1 case.

            # Calculate distances from center_p to all other existing '1's
            distances = []
            for j in range(M):
                if i == j: continue # Skip the one at center_p itself
                distances.append(abs(pos[j] - center_p))
            
            # Sort distances to easily find the smallest ones
            distances.sort()
            
            # Minimum cost calculation for this specific aliceIndex choice (pos[i])
            # We need target_k items. We can use 'c' Type 1 changes (cost 2 each)
            # and N_exist = target_k - c existing ones (cost = sum of distances).
            current_min_cost_for_p_i = float('inf')
            
            # Precompute prefix sums of sorted distances for efficient summation
            # `distances` array has M-1 elements. prefix_sum size M for 1-based indexing.
            dist_prefix_sum = [0] * M 
            for D_idx in range(len(distances)):
                dist_prefix_sum[D_idx+1] = dist_prefix_sum[D_idx] + distances[D_idx]

            # Iterate through possible number of Type 1 changes 'c' from 0 to maxChanges
            for c in range(maxChanges + 1):
                # Number of existing ones needed
                N_exist = target_k - c
                
                if N_exist <= 0:
                    # If we need 0 or fewer existing ones (meaning c >= target_k).
                    # We can satisfy the need using 'target_k' changes. Total cost = 2 * target_k.
                    cost = 2 * target_k
                    current_min_cost_for_p_i = min(current_min_cost_for_p_i, cost)
                    # Further increase in 'c' won't decrease cost because N_exist remains <= 0. Break inner loop.
                    break 
                
                if N_exist > len(distances):
                    # Not enough existing ones available in the `distances` array for this 'c'.
                    # This scenario means we cannot collect target_k ones with c changes.
                    continue # Try next c value.

                # Cost = 2*c (for changes) + sum of N_exist smallest distances
                cost = 2 * c + dist_prefix_sum[N_exist]
                current_min_cost_for_p_i = min(current_min_cost_for_p_i, cost)
            
            # Update overall minimum moves found across all potential aliceIndex choices (pos[i])
            min_total_moves = min(min_total_moves, current_min_cost_for_p_i)

        # Return the overall minimum moves found.
        # The problem constraints guarantee a solution exists.
        return min_total_moves