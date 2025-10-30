# YOUR CODE HERE
import sys

def solve():
    """
    Solves the game theory problem using Sprague-Grundy theorem.
    The game is decomposed into N independent subgames, one for each integer A_i.
    The Grundy number (nim-value) for each subgame is calculated.
    The winner is determined by the XOR sum of the Grundy numbers.
    """
    N = int(sys.stdin.readline())
    # N is guaranteed >= 1 by constraints
    
    A = list(map(int, sys.stdin.readline().split()))

    MAX_A_val = 0
    # Find max value in A. Handles empty list case implicitly, although N>=1 ensures A is not empty.
    if A: 
      # Constraints state A_i >= 2, so max(A) will be at least 2 if N>=1.
      MAX_A_val = max(A)
    
    # Need to calculate Grundy numbers up to max(A_i).
    # Set minimum calculation range to 2, because the smallest possible A_i is 2.
    # If A contains only 2, MAX_A_val is 2. MAX_CALC will be 2.
    MAX_CALC = max(MAX_A_val, 2) 
    
    # Precompute proper divisors using a sieve-like method
    # divisors[k] will store a list of positive integers d such that d divides k and d < k.
    divisors = [[] for _ in range(MAX_CALC + 1)]
    
    # Iterate through potential divisors d
    for d in range(1, MAX_CALC + 1):
        # Iterate through multiples k = m*d, starting from m=2 because d is not a *proper* divisor of d.
        # The range goes up to MAX_CALC.
        for k in range(2 * d, MAX_CALC + 1, d):
            # Add d to the list of proper divisors for k
            divisors[k].append(d)

    # grundy[k] will store the Grundy number of the subgame starting with integer k.
    # grundy[0] and grundy[1] are 0. Initialize the array with zeros.
    grundy = [0] * (MAX_CALC + 1) 
    
    # Use a boolean array for finding mex (Minimum Excluded value) efficiently.
    # The maximum possible Grundy value is bounded by the maximum number of divisors.
    # For K <= 100000, the maximum number of divisors is 128.
    # So, the maximum Grundy value G(K) <= 128.
    # We use a slightly larger buffer size for safety.
    MAX_GRUNDY_POSSIBLE = 150 
    seen_mex_check = [False] * MAX_GRUNDY_POSSIBLE 

    # Calculate Grundy numbers iteratively from 2 up to MAX_CALC.
    for k in range(2, MAX_CALC + 1):
        
        # Store the distinct Grundy values encountered for divisors of k.
        # This is used to efficiently reset the seen_mex_check array later.
        indices_to_reset = [] 
        for d in divisors[k]:
            # Get the precomputed Grundy value of the divisor d.
            g_val = grundy[d]
            
            # Check if the Grundy value is within the expected range/buffer size.
            if g_val < MAX_GRUNDY_POSSIBLE:
                 # Mark this Grundy value as 'seen' only if it hasn't been marked yet for this k.
                 if not seen_mex_check[g_val]: 
                     seen_mex_check[g_val] = True # Mark it as seen
                     indices_to_reset.append(g_val) # Record index to reset later
            else:
                 # This case indicates the Grundy value exceeded the expected bound.
                 # Based on analysis, this should not happen for K <= 100000.
                 # In a competitive programming context, we usually assume the analysis is correct
                 # and constraints hold. No explicit error handling added here.
                 pass 
                 
        # Find the smallest non-negative integer g that was not seen (mex).
        g = 0
        # The check `g < MAX_GRUNDY_POSSIBLE` ensures we don't access the array out of bounds.
        while g < MAX_GRUNDY_POSSIBLE and seen_mex_check[g]: 
            g += 1
        
        # Optional check: If g reaches MAX_GRUNDY_POSSIBLE, it means mex computation failed due to buffer size limit.
        # Again, assumed not to happen under problem constraints.
        # if g == MAX_GRUNDY_POSSIBLE:
        #      # Handle error or debug print.
        #      pass

        # Store the computed Grundy number for k.
        grundy[k] = g
        
        # Reset the flags in seen_mex_check efficiently using the recorded indices.
        # This prepares the array for the next iteration (next k).
        for idx in indices_to_reset:
            seen_mex_check[idx] = False

    # Compute the XOR sum of Grundy numbers for the initial sequence A.
    # According to Sprague-Grundy theorem, this determines the winner.
    total_xor_sum = 0
    for val in A:
        # val is guaranteed to be >= 2 and <= MAX_A_val <= MAX_CALC by constraints and logic.
        total_xor_sum ^= grundy[val]

    # Determine the winner based on the total XOR sum.
    # If XOR sum > 0, the initial state is an N-position (non-zero Grundy value), so the first player (Anna) wins.
    # If XOR sum = 0, the initial state is a P-position (zero Grundy value), so the second player (Bruno) wins.
    if total_xor_sum > 0:
        print("Anna") 
    else:
        print("Bruno") 

# Ensure the script runs the solve function when executed directly.
if __name__ == '__main__':
    solve()