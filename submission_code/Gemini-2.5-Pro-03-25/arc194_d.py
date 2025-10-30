# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep structures typical in parenthesis problems.
# N can be up to 5000, the recursion depth could potentially reach N/2 = 2500.
# Standard Python limit is often 1000. Setting it higher is necessary.
try:
    # Set recursion depth limit slightly larger than N/2. 
    # 6000 should be safe for N=5000.
    sys.setrecursionlimit(6000) 
except Exception as e: 
     # This might fail on some systems/platforms depending on OS limits or configurations.
     # Printing a warning might be helpful for debugging but not required for typical contest setup.
     # print(f"Warning: Failed to set recursion depth limit: {e}", file=sys.stderr)
     # The program will continue with the default limit. If the test cases are deep, it might crash.
     pass

MOD = 998244353

# Precompute factorials and their modular inverses up to MAX_N
# N <= 5000, so MAX_N = 5001 is sufficient.
MAX_N = 5001 
fact = [1] * MAX_N
invfact = [1] * MAX_N

# Calculate factorials modulo MOD
for i in range(1, MAX_N):
    fact[i] = (fact[i - 1] * i) % MOD

# Calculate modular inverse of the largest factorial using Fermat's Little Theorem (since MOD is prime)
# pow(a, b, m) computes (a^b) % m efficiently.
invfact[MAX_N - 1] = pow(fact[MAX_N - 1], MOD - 2, MOD)

# Calculate modular inverses for smaller factorials iteratively using the relation:
# (k!)^{-1} = ((k+1)! * (k+1)^{-1})^{-1} = (k+1)!^{-1} * (k+1)
# So invfact[i] = invfact[i+1] * (i+1) % MOD
for i in range(MAX_N - 2, -1, -1):
    invfact[i] = (invfact[i + 1] * (i + 1)) % MOD

def multinomial_coeff(k, counts):
    """ 
    Computes the multinomial coefficient k! / (a1! * a2! * ... * am!) mod MOD.
    'k' is the total number of items (which is sum of counts).
    'counts' is a list or tuple containing [a1, a2, ..., am].
    """
    # Base case: if k=0 (empty sequence of components), the coefficient is 1.
    if k == 0:
      return 1

    # Start with k!
    res = fact[k]
    # Multiply by the modular inverse of each ai!
    for count in counts:
        # Basic validation: counts should be non-negative and within precomputed range.
        # This should hold true given the problem structure and N limit.
        if count < 0 or count >= MAX_N:
             # This indicates an unexpected state, possibly an error in logic or input handling.
             # For robustness, could raise an error or return 0. Assuming valid inputs.
             pass
        
        # Multiply by (count!)^{-1} mod MOD
        res = (res * invfact[count]) % MOD
    return res

# Memoization table (dictionary) to store results for computed substrings
memo = {}

def count_reachable(S):
    """ 
    Recursively computes the number of distinct strings reachable from string S
    using the specified operation, modulo MOD.
    """
    N = len(S)
    # Base case: An empty string represents a single state (itself).
    if N == 0:
        return 1
    
    # Check if the result for S is already memoized.
    # Using the string S directly as the key. Python's string hashing is efficient.
    if S in memo:
        return memo[S]

    # Check if S is of the form (A), where A is a valid parenthesis sequence.
    # This is true if S starts with '(', ends with ')', and the parenthesis balance
    # never drops to 0 strictly inside S (i.e., between index 1 and N-2).
    is_form_A = False
    # Check S[0] and S[-1] requires N > 0. For N=1 this check fails. But N must be even for valid sequence.
    # N=2 case: S = "()". N-1 = 1. Loop range(1) is empty. is_minimal_enclosure stays True. correct.
    if N > 0 and S[0] == '(' and S[-1] == ')':
        bal = 0
        is_minimal_enclosure = True
        # Iterate through indices 0 to N-2.
        for i in range(N - 1): 
            if S[i] == '(':
                bal += 1
            else:
                bal -= 1
            
            # If balance drops to 0 or below *before* reaching the last character's position (index N-1),
            # it means S can be decomposed into XY... where X=S[0...i] is a valid sequence.
            # Therefore, S is not minimally enclosed by its outer parentheses. It's not of form (A).
            if bal <= 0 and i < N - 1: 
                 is_minimal_enclosure = False
                 break
        
        # If the loop completes without breaking, is_minimal_enclosure remains True.
        # This implies S is of the form (A).
        # The final balance check (bal == 0 after processing S[N-1]) is implicitly guaranteed
        # because S is given as a valid parenthesis sequence.
        if is_minimal_enclosure:
             is_form_A = True

    if is_form_A:
        # If S = (A), the number of reachable states is the same as for A.
        # Recursively call for the inner string A = S[1:-1].
        res = count_reachable(S[1:-1])
        # Store result in memo before returning.
        memo[S] = res
        return res

    # If S is not of the form (A), it must be a concatenation S = S1 S2 ... Sk
    # where each Si is a minimal non-empty valid parenthesis sequence.
    components = []
    bal = 0
    start_idx = 0
    # Scan S to find the minimal components based on balance returning to 0.
    for i in range(N):
        if S[i] == '(':
            bal += 1
        else:
            bal -= 1
        
        # When balance returns to 0, we have found a component S[start_idx...i].
        if bal == 0:
            components.append(S[start_idx : i + 1])
            # Update start index for the next component.
            start_idx = i + 1
    
    # Number of minimal components found.
    k = len(components)
    
    # Calculate P = Product[N(Si)] mod MOD, where N(Si) is the result for component Si.
    prod_counts = 1
    for comp in components:
         # Recursively find the count for each component.
         comp_res = count_reachable(comp)
         # Multiply into the running product P, taking modulo at each step.
         prod_counts = (prod_counts * comp_res) % MOD
         
    # Group identical components to find their counts [a1, a2, ..., am]
    # This is needed for the multinomial coefficient calculation.
    component_counts = {}
    for comp in components:
        # Using the component string itself as the key for grouping.
        component_counts[comp] = component_counts.get(comp, 0) + 1
    
    # Get the list of counts [a1, a2, ..., am].
    counts_list = list(component_counts.values())
    
    # Calculate the multinomial coefficient M = k! / (a1! * a2! * ... * am!) mod MOD.
    # This represents the number of distinct permutations of the components.
    M = multinomial_coeff(k, counts_list)
    
    # The final result for S is M * P mod MOD.
    final_res = (M * prod_counts) % MOD
    # Store result in memo before returning.
    memo[S] = final_res
    return final_res

# Main execution part:
# Read N and S from standard input.
N_input = int(sys.stdin.readline())
S_input = sys.stdin.readline().strip()

# Call the main recursive function to compute the result.
result = count_reachable(S_input)

# Print the final result to standard output.
print(result)