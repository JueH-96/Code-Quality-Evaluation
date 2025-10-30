import math
from collections import Counter

# Helper function compute_min_conv_abs_extended 
# Computes g(p) = min_j (f(j) + |p - j|) for p in range 0..N_carry-1
# Also returns boundary values needed for p outside this range:
# min_f_minus_j = min_j (f(j) - j)
# min_f_plus_j = min_j (f(j) + j)
# The algorithm runs in O(N_carry) time.
def compute_min_conv_abs_extended(f):
    """
    Computes the min convolution with absolute difference kernel |p-j|.
    Args:
        f: A list representing the function f(j) for j = 0..N_carry-1.
           f[j] can be float('inf') if the state is unreachable.
    Returns:
        A tuple (g, min_f_minus_j, min_f_plus_j):
        g: A list where g[p] = min_j (f(j) + |p-j|) for p = 0..N_carry-1.
        min_f_minus_j: The minimum value of f(j) - j over all j.
        min_f_plus_j: The minimum value of f(j) + j over all j.
    """
    N_carry = len(f)
    if N_carry == 0: return [], float('inf'), float('inf') 

    # Forward pass: Compute g_fwd(p) = min_{q<=p} (f(q) - q)
    # g_fwd represents the running minimum of f(q)-q up to index p.
    g_fwd = [float('inf')] * N_carry
    current_min_fmj = float('inf') # Renamed from current_min for clarity
    for p in range(N_carry):
        # Check if f[p] is a valid (finite) cost
        if p < len(f) and f[p] != float('inf'):
            val_f_minus_p = f[p] - p
            current_min_fmj = min(current_min_fmj, val_f_minus_p)
        g_fwd[p] = current_min_fmj
    # The overall minimum of f(j)-j is the last value computed.
    min_f_minus_j = current_min_fmj 

    # Backward pass: Compute g_bwd(p) = min_{q>=p} (f(q) + q)
    # g_bwd represents the running minimum of f(q)+q from index p to the end.
    g_bwd = [float('inf')] * N_carry
    current_min_fpj = float('inf') # Renamed from current_min
    for p in range(N_carry - 1, -1, -1):
         # Check if f[p] is valid
        if p < len(f) and f[p] != float('inf'):
             val_f_plus_p = f[p] + p
             current_min_fpj = min(current_min_fpj, val_f_plus_p)
        g_bwd[p] = current_min_fpj
    # The overall minimum of f(j)+j is the first value computed in backward pass.
    min_f_plus_j = g_bwd[0] if N_carry > 0 else float('inf')

    # Combine: g(p) = min(p + g_fwd[p], -p + g_bwd[p])
    # This calculates min_j (f(j) + |p-j|) efficiently.
    g = [float('inf')] * N_carry
    for p in range(N_carry):
        term1 = float('inf')
        # Check if g_fwd[p] is valid
        if p < len(g_fwd) and g_fwd[p] != float('inf'):
            term1 = p + g_fwd[p] # Corresponds to min_{q<=p} f(q) + p-q
        term2 = float('inf')
        # Check if g_bwd[p] is valid
        if p < len(g_bwd) and g_bwd[p] != float('inf'):
            term2 = -p + g_bwd[p] # Corresponds to min_{q>=p} f(q) + q-p
        
        g[p] = min(term1, term2)

    return g, min_f_minus_j, min_f_plus_j


class Solution:
    """
    Solves the problem of finding the minimum operations to make a string "good".
    A string is good if all characters occur the same number of times.
    Operations: delete, insert, change c -> c+1. Each costs 1.
    """
    def makeStringGood(self, s: str) -> int:
        """
        Calculates the minimum operations using dynamic programming over characters 'a' through 'z'
        and iterating through possible target frequencies 'k'. Optimized with min-convolution.
        Args:
            s: The input string.
        Returns:
            The minimum number of operations.
        """
        
        n = len(s)
        # According to constraints N >= 3.

        # Precompute character counts
        counts = Counter(s)
        initial_counts = [0] * 26
        for i in range(26):
            char = chr(ord('a') + i)
            initial_counts[i] = counts[char]

        # Initialize minimum cost found so far. Maximum possible cost is n (deleting all chars).
        min_total_cost = n 
        # Maximum carry needed could be up to n. The DP state size is n+1 (for carry 0 to n).
        MaxCarry = n 

        # Iterate through possible target frequencies k.
        # The optimal k is likely bounded. Testing up to N+1 seems sufficient based on analysis.
        # Cost for k=0 (empty string) is N, already accounted for in min_total_cost initialization.
        for k in range(1, n + 2): 
            
            # dp_curr[c] stores the minimum cost to process prefix of alphabet up to previous character,
            # ending with 'c' characters carried over (changed to next char).
            dp_curr = [float('inf')] * (MaxCarry + 1)
            # Base case: Before processing 'a', cost is 0 with 0 carry.
            dp_curr[0] = 0 

            # Dynamic programming over alphabet characters i = 0 ('a') to 25 ('z')
            for i in range(26): 
                count_i = initial_counts[i] # Initial count of character 'a'+i
                
                # Compute g(p) = min_j (dp_curr[j] + |p-j|) using the optimized O(N) function.
                # Also get boundary minimums needed for p outside [0, MaxCarry].
                g, min_fmj, min_fpj = compute_min_conv_abs_extended(dp_curr)

                # dp_next[c] will store the minimum cost after processing character 'a'+i,
                # ending with 'c' characters carried over.
                dp_next = [float('inf')] * (MaxCarry + 1)
                
                # Calculate dp_next[carry_out] for all possible carry_out values 0..MaxCarry
                for carry_out in range(MaxCarry + 1):
                    
                    # Calculate cost if target count for char 'a'+i is 0
                    p0 = carry_out - count_i # Effective position for g function query
                    g_p0 = float('inf')
                    # Determine value of g(p0) using calculated g array and boundary values
                    if 0 <= p0 <= MaxCarry:
                        if p0 < len(g): # Ensure index is valid
                           g_p0 = g[p0]
                    elif p0 < 0: # p0 is outside the range [0, MaxCarry], use boundary formula
                         if min_fpj != float('inf'):
                            # Formula for p < 0: g(p) = -p + min_j(f(j)+j)
                            g_p0 = -p0 + min_fpj 
                    else: # p0 > MaxCarry, use boundary formula
                         if min_fmj != float('inf'):
                            # Formula for p > N: g(p) = p + min_j(f(j)-j)
                            g_p0 = p0 + min_fmj 
                    
                    cost0 = float('inf')
                    if g_p0 != float('inf'):
                           # Total cost for this path = previous min cost + transition cost
                           # Transition cost = abs(Target + carry_out - count_i - carry_in) + carry_out
                           # The optimized calculation gives: g(Target + carry_out - count_i) + carry_out
                           cost0 = g_p0 + carry_out
                    
                    # Calculate cost if target count for char 'a'+i is k
                    pk = k + carry_out - count_i # Effective position for g function query
                    g_pk = float('inf')
                    # Determine value of g(pk)
                    if 0 <= pk <= MaxCarry:
                         if pk < len(g): # Ensure index is valid
                            g_pk = g[pk]
                    elif pk < 0: # pk is outside the range [0, MaxCarry]
                         if min_fpj != float('inf'):
                            g_pk = -pk + min_fpj
                    else: # pk > MaxCarry
                         if min_fmj != float('inf'):
                            g_pk = pk + min_fmj
                            
                    costk = float('inf')
                    if g_pk != float('inf'):
                             # Total cost = g(pk) + carry_out
                             costk = g_pk + carry_out

                    # The minimum cost to reach state (i, carry_out) is the minimum of
                    # choosing target=0 or target=k for character 'a'+i.
                    dp_next[carry_out] = min(cost0, costk)

                # Update DP state for the next character iteration
                dp_curr = dp_next

            # After processing all 26 characters for frequency k, the final minimum cost
            # must end with 0 carry (no characters changed beyond 'z').
            # Check if dp_curr[0] is finite (meaning a valid sequence of operations exists)
            if dp_curr[0] != float('inf'): 
                 min_total_cost = min(min_total_cost, dp_curr[0])

        # Return the overall minimum cost found across all tested k values.
        return min_total_cost