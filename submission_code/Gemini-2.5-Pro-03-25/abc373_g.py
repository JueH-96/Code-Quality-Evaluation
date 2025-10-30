# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep recursion
# N can be up to 300, the recursion depth could reach N.
# Python's default limit is often 1000.
# Setting it higher (e.g., 2000) provides a safety margin for N=300 cases.
sys.setrecursionlimit(2000) 

def solve():
    N = int(sys.stdin.readline())
    
    # Read point coordinates and store their original 0-based index along with coordinates.
    # P[i] = [A_i, B_i, i] where i is the original index (0 to N-1).
    # Q[i] = [C_i, D_i, i] where i is the original index (0 to N-1).
    P = []
    for i in range(N):
        P.append(list(map(int, sys.stdin.readline().split())) + [i]) 
    
    Q = []
    for i in range(N):
        Q.append(list(map(int, sys.stdin.readline().split())) + [i]) 

    # Computes the cross product (B-A) x (C-A) for 3 points A, B, C.
    # The sign determines orientation:
    # > 0: C is to the left of the directed line A -> B (counter-clockwise turn)
    # < 0: C is to the right of the directed line A -> B (clockwise turn)
    # = 0: C is collinear with A and B. The problem statement guarantees no three distinct points are collinear,
    #     so this function will return non-zero for distinct input points A, B, C.
    def cross_product(pA, pB, pC):
        # The points pA, pB, pC are lists like [x, y, original_index]. We only need x, y coordinates.
        return (pB[0] - pA[0]) * (pC[1] - pA[1]) - (pB[1] - pA[1]) * (pC[0] - pA[0])

    # Memoization cache to store results of subproblems.
    # Keys are tuples representing the state: (tuple_of_P_indices, tuple_of_Q_indices)
    # Values are the resulting matching dictionary, or None if no solution exists for that state.
    memo = {}

    # Helper function to create a canonical key for the memoization cache.
    # Uses sorted tuples of indices to ensure that the order of points in the input tuples doesn't affect the key.
    def get_key(P_subset_indices, Q_subset_indices):
         return (tuple(sorted(P_subset_indices)), tuple(sorted(Q_subset_indices)))

    # Recursive function to find a non-crossing matching for the given subsets of P and Q points.
    # Input:
    #   P_subset_indices: tuple of original indices of P points in the current subproblem.
    #   Q_subset_indices: tuple of original indices of Q points in the current subproblem.
    # Output:
    #   A dictionary mapping original P index `p_idx` to original Q index `q_idx` for the matched pairs,
    #   or None if no non-crossing matching exists for this subproblem.
    def find_matching(P_subset_indices, Q_subset_indices):
        
        # Base case: If there are no P points (and thus no Q points), the empty matching is valid.
        if not P_subset_indices:
            return {} 

        # Create the canonical state key for memoization.
        state_key = get_key(P_subset_indices, Q_subset_indices)
        
        # Check if the result for this state is already cached.
        if state_key in memo:
             return memo[state_key] # Return the cached result (could be a dict or None).

        # Select a pivot P point. To ensure deterministic behavior, we pick the P point
        # with the minimum original index among the points in the current subset.
        pivot_P_idx = min(P_subset_indices)
        # Retrieve the pivot point's data: [X coordinate, Y coordinate, original index].
        P_current = P[pivot_P_idx] 

        # Create a tuple of indices for the remaining P points (excluding the pivot).
        remaining_P_indices = tuple(idx for idx in P_subset_indices if idx != pivot_P_idx)

        # Iterate through all possible Q points in the current subset to match with the pivot P point.
        for k_idx in Q_subset_indices:
            # Retrieve the current Q point's data.
            Q_current = Q[k_idx] 
            
            # Prepare lists to store indices of points partitioned by the line through P_current and Q_current.
            # L denotes "Left", R denotes "Right" relative to the directed segment P_current -> Q_current.
            P_L_indices = []
            Q_L_indices = []
            P_R_indices = []
            Q_R_indices = []
            
            # Partition the remaining P points based on their position relative to the line P_current -> Q_current.
            for p_idx in remaining_P_indices:
                P_check = P[p_idx]
                cp = cross_product(P_current, Q_current, P_check)
                if cp > 0: # P_check is Left
                    P_L_indices.append(p_idx)
                else: # cp < 0 implies P_check is Right (guaranteed by no-collinearity constraint).
                    P_R_indices.append(p_idx)

            # Partition the remaining Q points (excluding the matched Q point k_idx).
            remaining_Q_indices = tuple(idx for idx in Q_subset_indices if idx != k_idx)
            for q_idx in remaining_Q_indices:
                Q_check = Q[q_idx]
                cp = cross_product(P_current, Q_current, Q_check)
                if cp > 0: # Q_check is Left
                    Q_L_indices.append(q_idx)
                else: # cp < 0 implies Q_check is Right.
                    Q_R_indices.append(q_idx)
            
            # Check the balance condition: The number of P points on the Left side must equal
            # the number of Q points on the Left side. This is a necessary condition derived from
            # the non-crossing property. If it holds, the balance on the Right side also holds implicitly.
            if len(P_L_indices) == len(Q_L_indices): 
                
                # Recursively call find_matching for the Left subset of points.
                # Pass indices as tuples.
                res_L = find_matching(tuple(P_L_indices), tuple(Q_L_indices))
                
                # If the recursive call for the Left subset fails (returns None),
                # this choice of Q point (k_idx) is invalid. Continue to the next Q point.
                if res_L is None:
                    continue 

                # Recursively call find_matching for the Right subset of points.
                res_R = find_matching(tuple(P_R_indices), tuple(Q_R_indices))
                
                # If the recursive call for the Right subset fails (returns None),
                # this choice of Q point (k_idx) is invalid. Continue to the next Q point.
                if res_R is None:
                    continue
                
                # SUCCESS! Both recursive calls succeeded. A valid non-crossing matching is found
                # for this branch. Combine the results from the Left and Right subproblems
                # with the current match (pivot_P_idx -> k_idx).
                final_matching = {pivot_P_idx: k_idx}
                final_matching.update(res_L) # Add matches from the left subproblem
                final_matching.update(res_R) # Add matches from the right subproblem
                
                # Cache the successful result for this state before returning.
                memo[state_key] = final_matching
                return final_matching

        # If the loop completes without finding any valid Q point match for the pivot P point,
        # it means no non-crossing matching exists for this state.
        # Cache the failure (None) and return None.
        memo[state_key] = None 
        return None

    # Start the process by calling find_matching with all initial P and Q points (indices 0 to N-1).
    initial_P_indices = tuple(range(N))
    initial_Q_indices = tuple(range(N))
    
    final_result = find_matching(initial_P_indices, initial_Q_indices)

    # Check the final result.
    if final_result is None:
        # If final_result is None, no non-crossing matching exists for the entire problem.
        print("-1")
    else:
        # A non-crossing matching was found. final_result is a dictionary mapping
        # P's original 0-based index `p_idx` to Q's original 0-based index `q_idx`.
        
        # The problem asks for output R = (R_1, R_2, ..., R_N), where segment i connects
        # P_i and Q_{R_i} (using 1-based indexing for points P_i and Q_j in the problem statement).
        # Our 0-based P point `p_idx` corresponds to the problem's P_{p_idx + 1}.
        # Our 0-based Q point `q_idx` corresponds to the problem's Q_{q_idx + 1}.
        # The matching `final_result[p_idx] = q_idx` means P_{p_idx + 1} is matched with Q_{q_idx + 1}.
        # According to the problem statement, P_i connects to Q_{R_i}.
        # So, for i = p_idx + 1, we have R_i = q_idx + 1.
        # This means R_{p_idx + 1} = final_result[p_idx] + 1.
        
        # Construct the output list `output_R` where `output_R[i]` stores R_{i+1}.
        output_R = [0] * N
        for p_idx in range(N):
             # Calculate R_{p_idx + 1} using the matched Q index from final_result.
             output_R[p_idx] = final_result[p_idx] + 1
        
        # Print the result R_1 R_2 ... R_N separated by spaces.
        print(*(output_R))

# Call the main solver function to run the program.
solve()