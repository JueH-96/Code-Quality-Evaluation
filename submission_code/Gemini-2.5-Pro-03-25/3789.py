import collections
from typing import List

class Solution:
    """
    Calculates the maximum number of valid subarrays possible after removing exactly one conflicting pair.
    A subarray is valid if it does not contain both elements a and b for any remaining conflicting pair [a, b].
    The input array nums contains numbers from 1 to n in order.
    """
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        
        # If there are no conflicting pairs, all N*(N+1)/2 subarrays are valid.
        # N = n
        if not conflictingPairs:
            # The total number of non-empty subarrays is the sum 1 + 2 + ... + n = n*(n+1)/2
            return n * (n + 1) // 2

        k = len(conflictingPairs) # Number of conflicting pairs
        
        # Preprocess conflicting pairs: Store pairs by their activation index R = v-1.
        # A pair [a, b] is represented as [u, v] where u = min(a, b), v = max(a, b).
        # The pair introduces a constraint for subarrays nums[L..R] where L <= u-1 and R >= v-1.
        # This constraint becomes active (relevant) when the subarray's right endpoint R reaches v-1.
        # Each entry stores the constraint value u-1 and the original pair index i = 0..k-1.
        pairs_by_v = collections.defaultdict(list)
        for i in range(k):
            a, b = conflictingPairs[i]
            # Constraints state a != b.
            u, v = min(a, b), max(a, b)
            u_idx, v_idx = u - 1, v - 1 # Convert to 0-based indices.
            
            # Check if v is a valid element index (v >= 1 means v_idx >= 0).
            if v_idx >= 0: 
                 pairs_by_v[v_idx].append({'u_idx': u_idx, 'pair_idx': i})

        # Precompute activation info for each R = 0..n-1.
        # For each R, determine the maximum (m) and second maximum (m_prime) u_idx values
        # among pairs activating *exactly* at R. Also store the sets of pair indices (I, I_prime) 
        # achieving these maximum and second maximum values respectively.
        activations = []
        for R in range(n):
            # Initialize activation info for this R. Default is no new activations.
            activation_info = {'m': -1, 'I': set(), 'm_prime': -1, 'I_prime': set()}
            if R in pairs_by_v:
                current_pairs = pairs_by_v[R]
                # Group pairs activating at R by their constraint value u_idx to efficiently find top two distinct values.
                groups = collections.defaultdict(list)
                valid_u_indices = [] # Keep track of valid u_idx (>=0) encountered for this R.
                for p_info in current_pairs:
                    u_idx = p_info['u_idx']
                    pair_idx = p_info['pair_idx']
                    # Ensure u >= 1 (so u_idx >= 0). This indicates a valid constraint starting point.
                    if u_idx >= 0:
                         groups[u_idx].append(pair_idx)
                         valid_u_indices.append(u_idx)
                
                if valid_u_indices: # If any valid pairs activate at R
                    # Find unique u_idx values and sort them descending to identify max and second max.
                    unique_u_indices = sorted(list(set(valid_u_indices)), reverse=True)
                    
                    # Store max u_idx and its associated pair indices.
                    if len(unique_u_indices) > 0:
                        activation_info['m'] = unique_u_indices[0]
                        activation_info['I'] = set(groups[unique_u_indices[0]])
                    
                    # Store second max u_idx and its associated pair indices if it exists.
                    if len(unique_u_indices) > 1:
                        activation_info['m_prime'] = unique_u_indices[1]
                        activation_info['I_prime'] = set(groups[unique_u_indices[1]])
            activations.append(activation_info)

        # Calculate the base number of valid subarrays (with all conflicting pairs present).
        # Simultaneously calculate Delta_i for each pair i: the total increase in valid subarrays 
        # over all R if pair i were removed.
        delta = [0] * k # Initialize increase for each pair removal to 0.
        base_valid_count = 0 # Initialize base valid count.
        
        # State variables track the overall maximum and second maximum u_idx constraint active up to index R.
        # M_R defines the strongest lower bound constraint on L for subarrays ending at R: L must be > M_R.
        M_R = -1  # Max u_idx constraint active up to R. Initialized to -1 (representing no constraint).
        MaxIdx_R = set() # Set of pair indices that achieve M_R.
        M_prime_R = -1 # Second max u_idx constraint active up to R.
        MaxIdx_prime_R = set() # Set of pair indices that achieve M_prime_R.

        for R in range(n):
            # Get state (max constraints and indices) from previous step R-1.
            prev_M, prev_MaxIdx = M_R, MaxIdx_R
            prev_M_prime, prev_MaxIdx_prime = M_prime_R, MaxIdx_prime_R
            
            # Get activation info for the current step R.
            curr_act = activations[R]
            curr_m, curr_I = curr_act['m'], curr_act['I']
            curr_m_prime, curr_I_prime = curr_act['m_prime'], curr_act['I_prime']

            # Combine previous state and current activations to determine the new state for R.
            # We consider all candidate values {prev_M, prev_M_prime, curr_m, curr_m_prime}
            # and their associated indices to find the new top two distinct maximum constraint values active up to R.
            
            candidates = []
            # Add constraints from previous state if they are valid (represented by value >= 0).
            if prev_M != -1: candidates.append((prev_M, prev_MaxIdx))
            if prev_M_prime != -1: candidates.append((prev_M_prime, prev_MaxIdx_prime))
            # Add constraints from current activation if they are valid.
            if curr_m != -1: candidates.append((curr_m, curr_I))
            if curr_m_prime != -1: candidates.append((curr_m_prime, curr_I_prime))

            # Group candidates by constraint value (u_idx) and merge the sets of pair indices.
            val_to_indices = collections.defaultdict(set)
            distinct_vals = set()
            for val, indices in candidates:
                 val_to_indices[val].update(indices)
                 distinct_vals.add(val)

            # Sort the distinct constraint values found in descending order.
            sorted_distinct_vals = sorted(list(distinct_vals), reverse=True)

            # Update the state (M_R, MaxIdx_R, M_prime_R, MaxIdx_prime_R) for index R based on combined info.
            if len(sorted_distinct_vals) > 0:
                M_R = sorted_distinct_vals[0]
                MaxIdx_R = val_to_indices[M_R]
            else:
                 # If no constraints active up to R, M_R remains -1.
                 M_R = -1
                 MaxIdx_R = set()

            if len(sorted_distinct_vals) > 1:
                 M_prime_R = sorted_distinct_vals[1]
                 MaxIdx_prime_R = val_to_indices[M_prime_R]
            else:
                 # If less than two distinct constraints active, M_prime_R is -1.
                 M_prime_R = -1
                 MaxIdx_prime_R = set()

            # Calculate the number of valid subarrays ending at R based on the strongest constraint M_R.
            # A subarray nums[L..R] is valid with respect to all pairs if its start index L > M_R.
            # Valid L must satisfy M_R < L <= R. The number of such valid L is R - M_R.
            base_valid_count += (R - M_R)
            
            # Check if the maximum constraint M_R is imposed by exactly one pair i.
            # If so, removing this pair i would potentially increase the number of valid subarrays ending at R.
            # The increase for this specific R would be M_R - M_prime_R, because M_prime_R would become the new effective maximum constraint.
            if len(MaxIdx_R) == 1:
                # Get the unique pair index i. Since MaxIdx_R has size 1, list(MaxIdx_R)[0] safely extracts it.
                i = list(MaxIdx_R)[0] 
                # Add the potential increase to Delta_i. Delta[i] accumulates potential increases over all R.
                # M_R >= M_prime_R is guaranteed by definition, so the difference is non-negative.
                delta[i] += (M_R - M_prime_R) 

        # The final result is the base number of valid subarrays plus the maximum possible increase (max(Delta_i))
        # achieved by removing the single best conflicting pair.
        max_delta = 0
        if delta: # Check if delta list is non-empty (i.e., k > 0).
            max_delta = max(delta) # Find the maximum total increase achievable by removing one pair. Since delta values are non-negative, this is max(0, max(delta)).
        
        # Return the maximum possible number of valid subarrays.
        return base_valid_count + max_delta