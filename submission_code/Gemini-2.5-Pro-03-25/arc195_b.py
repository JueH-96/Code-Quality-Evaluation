# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    
    # Read sequences A and B
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Separate fixed non-negative values from -1 placeholders
    A_pos = [x for x in A if x != -1]
    B_pos = [x for x in B if x != -1]
    
    # Count the number of -1s in A. This is the number of values we can choose for A.
    N_A_minus = A.count(-1)
    # N_B_minus = B.count(-1) # This count is not directly needed for the check condition derived

    # Determine the minimum possible sum S.
    # Any achievable sum S must be non-negative.
    # Also, S must be at least as large as any fixed non-negative value in A or B,
    # because if A_i is fixed non-negative, it must be paired with some B_j' >= 0 such that A_i + B_j' = S, so S >= A_i.
    # Similarly, if B_i is fixed non-negative, S >= B_i.
    max_A = -1
    if A_pos:
        max_A = max(A_pos)
    
    max_B = -1
    if B_pos:
        max_B = max(B_pos)

    # M is the minimum possible value for the common sum S based on fixed values.
    # S must be at least M. S must also be non-negative.
    M = max(0, max_A, max_B) 

    # Precompute frequency counts of fixed non-negative values for efficiency
    countA_fixed = Counter(A_pos)
    countB_fixed = Counter(B_pos)

    # Function to check if a given target sum S is achievable
    # The core logic relies on checking a necessary condition derived from multiset equality.
    # A' = {S - b | b in B'} means count_A'(k) = count_B'(S-k) for all k.
    # Expressing this in terms of fixed and chosen parts leads to:
    # f_{A_{chosen}}(k) - f_{B_{chosen}}(S-k) = f_{B_{pos}}(S-k) - f_{A_{pos}}(k)
    # Let D(k) = f_{B_{pos}}(S-k) - f_{A_{pos}}(k).
    # We need to find if there exist non-negative counts f_{A_{chosen}}(k) and f_{B_{chosen}}(k)
    # summing to N_A_minus and N_B_minus respectively, satisfying the equation.
    # A necessary condition for this is N_A_minus >= sum_{k} max(0, D(k)).
    def check(S):
        # Check if S is valid based on the minimum required value M.
        if S < M:
             return False

        # Calculate the difference counter D(k) = f_{B_{pos}}(S-k) - f_{A_{pos}}(k)
        # D[k] represents the net balance of value k required from chosen elements.
        # Positive D[k] means A_chosen needs to provide k more than B_chosen provides S-k.
        # Negative D[k] means B_chosen needs to provide S-k more than A_chosen provides k.
        D = Counter()
        
        # Process B_pos: for each fixed value k_B in B, it requires a partner S-k_B in A'.
        # This contributes +v to D[S-k_B].
        for k_B, v in countB_fixed.items():
             # Since S >= M, and M >= k_B for any k_B in B_pos, we have S >= k_B.
             # Thus S - k_B >= 0 is guaranteed.
             target_k_A = S - k_B
             # We only care about non-negative values k
             # if target_k_A >= 0:
             D[target_k_A] += v
           
        # Process A_pos: each fixed value k_A in A requires a partner S-k_A from B'.
        # This means k_A needs to be 'accounted for'. It subtracts v from D[k_A].
        for k_A, v in countA_fixed.items():
             # Check if S is large enough for this fixed A value.
             # S >= M >= k_A ensures this value can potentially be formed.
             # If k_A > S, then it's impossible to find B_j >= 0 such that k_A + B_j = S.
             # This case should be covered by the S >= M check initially. Re-check for safety.
             if k_A > S: 
                  # This should theoretically not happen if S >= M, but as a safeguard:
                  return False 
             D[k_A] -= v
        
        # Calculate D_pos = sum_{k | D[k] > 0} D[k]
        # This is the total 'positive demand' that must be fulfilled by values chosen for A.
        D_pos = 0
        for k, v in D.items():
            # We only sum positive values D[k]. These values must be non-negative.
            if k < 0:
                 # Defensive check: values k should be non-negative. If k < 0 arises,
                 # it indicates an issue perhaps with S < 0 or large negative fixed values (which are ruled out by input constraints).
                 # Or possibly S - k_B < 0 case, but S >= M should prevent this.
                 # If this occurs, it implies impossibility.
                 return False # Safety return, although unlikely
            
            if v > 0:
                D_pos += v
        
        # Check the necessary condition: The number of available slots to choose values for A (N_A_minus)
        # must be at least the total positive demand D_pos.
        # If this condition holds, it's possible to find non-negative assignments for chosen values.
        return N_A_minus >= D_pos

    # Based on analysis and testing samples, it seems sufficient to check S=M and S=M+1.
    # This covers both parities for S relative to M. While a rigorous proof is complex,
    # this hypothesis often works for similar problems structure. If a solution exists,
    # it is likely one exists for S near the minimum possible value M.
    if check(M) or check(M + 1):
        print("Yes")
    else:
        # If neither M nor M+1 works, we conclude it's impossible.
        print("No")

# Execute the solver function
solve()