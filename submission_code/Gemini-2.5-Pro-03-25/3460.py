import sys
from typing import List

# Set a reasonable recursion depth limit, although this solution is iterative.
# sys.setrecursionlimit(2000) 

MOD = 10**9 + 7

class Solution:
    """
    Solves the Number of Permutations problem using dynamic programming.
    The problem asks for the number of permutations of [0, 1, ..., n-1] that satisfy
    certain requirements on the number of inversions in their prefixes.

    The core idea is to build the permutations iteratively by considering permutations
    of length k based on permutations of length k-1. We use dynamic programming
    to keep track of the number of permutations with a specific number of inversions.

    Let dp[k][inv] be the number of permutations of [0, ..., k-1] with exactly 'inv' inversions.
    When we construct permutations of length k from permutations of length k-1, we insert the element (k-1)
    into permutations of [0, ..., k-2]. Inserting (k-1) at position j (0-indexed) adds (k-1 - j) inversions.
    The number of added inversions ranges from 0 (inserting at the end) to k-1 (inserting at the beginning).
    The transition is: dp[k][inv] = sum(dp[k-1][inv - i]) for i from 0 to k-1.

    We optimize space by using a 1D DP array, dp[inv], representing the counts for the current length k.
    Prefix sums are used to efficiently compute the transitions in O(M) time per step, where M is the max inversion count tracked.

    The constraints state that required inversion counts cnt_i are at most 400. We argue that we only need
    to track inversion counts up to M=400 because any permutation path exceeding M inversions cannot satisfy
    any future requirement (since all required counts are <= M and inversions only increase).
    
    Requirements are applied at each step k: if a requirement exists for end_i = k-1, we filter the dp state
    to keep only the counts for permutations satisfying that specific requirement.
    """
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        
        # The maximum inversion count specified in requirements is at most 400.
        # We only need to track inversion counts up to this maximum M.
        M = 400 
        
        # Convert requirements list to a dictionary for efficient O(1) lookup.
        # req_map maps end_index (k-1) to the required inversion count cnt_i.
        req_map = {end_i: cnt_i for end_i, cnt_i in requirements}

        # Initialize DP array. dp[inv] will store the number of permutations 
        # of length k having exactly inv inversions.
        # Base case: k=1. The only permutation is [0], which has length 1 and 0 inversions.
        dp = [0] * (M + 1)
        dp[0] = 1 

        # Apply requirement for k=1 (corresponding to end_i=0) if specified.
        if 0 in req_map:
            target_inv = req_map[0]
            if target_inv != 0:
                # A permutation of length 1 must have 0 inversions.
                # If the requirement specified is non-zero, it's impossible to satisfy.
                return 0
            # If target_inv is 0, the dp state [1, 0, ..., 0] is already correct, no change needed.

        # Iterate from k=2 up to n, building permutations length by length.
        for k in range(2, n + 1):
            
            # Compute prefix sums of the current dp array (which represents state for length k-1).
            # ps[inv] = sum(dp[i] for i=0..inv) % MOD
            ps = [0] * (M + 1)
            current_sum = 0
            for inv in range(M + 1):
                current_sum = (current_sum + dp[inv]) % MOD
                ps[inv] = current_sum

            # Compute the new DP state (for length k) into new_dp array.
            new_dp = [0] * (M + 1)
            for inv in range(M + 1):
                # The number of permutations of length k with inv inversions is derived from permutations of length k-1.
                # Transition: dp[k][inv] = sum(dp[k-1][inv - i] for i in range(k)).
                # Using prefix sums: sum = ps[inv] - ps[inv - k] (handle index < 0).
                
                # Calculate the index for the subtrahend ps[inv - k].
                lower_bound_idx = inv - k
                
                # Get the value ps[inv - k]. If index is negative, its contribution is 0.
                subtrahend = ps[lower_bound_idx] if lower_bound_idx >= 0 else 0
                
                # Calculate new_dp[inv], ensuring the result remains non-negative modulo MOD.
                new_dp[inv] = (ps[inv] - subtrahend + MOD) % MOD
            
            # Update the DP array to reflect the state for length k.
            dp = new_dp

            # Apply requirement for current length k (corresponding to end_i = k - 1) if specified.
            if k - 1 in req_map:
                target_inv = req_map[k - 1]
                
                # Check if the required inversion count is theoretically possible for a permutation of length k.
                # The maximum possible number of inversions for length k is k * (k - 1) / 2.
                max_possible_inv = k * (k - 1) // 2
                if target_inv > max_possible_inv:
                     # Requirement is impossible to satisfy. No permutations can exist.
                     return 0 

                # Apply the requirement filter based on target_inv relative to M:
                if target_inv > M:
                    # If required count exceeds M, our DP state doesn't track this high.
                    # Since all tracked states have inversion count <= M, none can satisfy this requirement.
                    # Zero out the entire dp array because no paths are valid anymore.
                    for inv_idx in range(M + 1):
                         dp[inv_idx] = 0
                    # Further computations will correctly propagate zeros.
                else:
                    # Requirement target_inv is within the tracked range [0, M].
                    # Filter the dp array: only permutations with exactly target_inv inversions are valid moving forward.
                    # Keep the count dp[target_inv], set all other entries to 0.
                    target_val = dp[target_inv]
                    for inv_idx in range(M + 1):
                        dp[inv_idx] = 0
                    dp[target_inv] = target_val

        # After the loop completes (k=n), the dp array contains counts for permutations of length n
        # that satisfy all specified requirements.
        # The problem guarantees a requirement for end_i = n - 1 exists.
        # This final requirement was applied in the last iteration (k=n).
        # The final answer is the total count of valid permutations.
        # This is the sum of all entries in the final dp array.
        # Since the final requirement filter was applied, the dp array either is all zeros 
        # (if requirement was impossible or > M) or has a single non-zero entry at dp[final_target_inv]. 
        # Summing the dp array correctly computes the result in all cases.
        total_count = sum(dp) % MOD
        return total_count