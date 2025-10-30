# YOUR CODE HERE
import sys

# Increase recursion depth limit for potentially deep search paths
# The number of divisors of N can be large for N <= 10^12, potentially exceeding default limit.
try:
    # Set a generous recursion depth limit. Maximum number of divisors is around 13440.
    # Setting to 20000 should be safe.
    sys.setrecursionlimit(20000) 
except Exception as e:
    # In some environments changing recursion depth might be restricted.
    # Silently ignore or print a warning. For competitive programming, usually allowed.
    # print(f"Warning: Could not set recursion depth: {e}", file=sys.stderr)
    pass 

# Memoization dictionary stores results for solve(target_N)
# Key: target_N (an integer divisor of the original N)
# Value: tuple (found_flag, P_list_suffix, q_str) indicating if a decomposition was found for target_N,
#        the list of P strings forming the outer layers for target_N's decomposition,
#        and the string representation of the central palindrome Q.
memo = {} 

def is_palindrome(s):
    """Helper function to check if a string is a palindrome."""
    return s == s[::-1]

def check_Q(q_val):
    """
    Checks if an integer q_val can be represented as a palindrome string Q
    that satisfies the problem constraints: consisting only of digits '1'-'9'.
    Returns a tuple (is_valid, q_string_representation).
    """
    # Ensure q_val is positive. N >= 1, so divisors should be positive.
    if q_val <= 0: return False, ""

    s = str(q_val)
    # Condition: String must contain only digits '1'-'9'. Check for '0'.
    if '0' in s:
        return False, ""
    # Condition: String must be a palindrome.
    if not is_palindrome(s):
        return False, ""
    
    # If all conditions are met, return True and the string representation.
    return True, s

# Precompute candidate factors: products k = p * p' where p = val(P) and p' = val(reverse(P))
# P is a string of length 1 to 6 consisting of digits '1'-'9'.
candidates = []

# Generate P strings iteratively for lengths 1 to 6
current_p_strs = [str(d) for d in range(1, 10)] # Start with single digits (length 1)

for length in range(1, 7): # Process lengths 1 through 6
    next_p_strs = [] # To store strings for the next length level
    for p_str in current_p_strs:
        # Process the string p_str of the current length `length`
        p_val = int(p_str)
        rev_p_str = p_str[::-1]
        # The reversed string `rev_p_str` will also only contain digits '1'-'9'
        # because p_str only contains digits '1'-'9'.
        rev_p_val = int(rev_p_str)
        
        try:
             # Calculate the product k = p * p'
             k = p_val * rev_p_val
        except OverflowError:
             # This is extremely unlikely in standard Python with arbitrary precision integers.
             # Included as a safeguard.
             continue 

        # We only consider factors k > 1. k=1 corresponds to P='1', which doesn't help decompose N.
        if k > 1: 
             # Store the pair (k, P_string)
             candidates.append((k, p_str))
        
        # Generate strings for the next length level if the current length is less than 6
        if length < 6:
            for d in range(1, 10):
                 next_p_strs.append(p_str + str(d))
    
    # Update the list of strings for the next iteration
    current_p_strs = next_p_strs

# Sort candidates descending by the value of k. This is an optimization heuristic:
# trying larger factors first might lead to a solution faster or prune branches earlier.
candidates.sort(key=lambda x: x[0], reverse=True)

# This list will hold candidates filtered based on N (k <= N).
# It will be accessed by the recursive function `solve`.
current_candidates = []

# Recursive function using memoization to find a valid palindromic decomposition
def solve(target_N):
    """
    Recursively attempts to decompose target_N into factors of form k = p*p' and a central palindrome Q.
    Returns tuple (found_flag, P_list, q_str) based on memoization or computation.
    """
    
    # Check memoization table first to avoid recomputing
    if target_N in memo:
        return memo[target_N]

    # Base Case: Check if target_N itself can form the middle palindrome Q
    is_valid_q, q_str = check_Q(target_N)
    if is_valid_q:
        # If target_N is a valid Q, this is a successful base case.
        # The P_list component for this subproblem solution is empty because Q is the innermost part.
        result = (True, [], q_str)
        memo[target_N] = result
        return result

    # Recursive Step: Iterate through the filtered candidate factors
    for k, p_str in current_candidates: 
        # Optimization: if k > target_N, remaining candidates will also be > target_N
        # because the list is sorted descending by k. Skip further checks.
        if k > target_N: 
             continue 
        
        # Check if candidate factor k divides target_N
        if target_N % k == 0:
            new_target_N = target_N // k
            
            # Recursively call solve for the quotient new_target_N
            found_flag, P_suffix_list, q_str_from_rec = solve(new_target_N)
            
            # Check if the recursive call found a valid decomposition for new_target_N
            if found_flag:
                # A solution was found for the subproblem. Combine current P ('p_str') 
                # with the list returned from recursion (P_suffix_list).
                # `p_str` forms the next outer layer around the structure found for `new_target_N`.
                current_P_list = [p_str] + P_suffix_list
                
                result = (True, current_P_list, q_str_from_rec)
                memo[target_N] = result
                # Return the first solution found. This might not be the shortest/simplest,
                # but the problem asks for *any* valid string.
                return result 

    # If the loop finishes without finding any factor k that leads to a valid solution
    result = (False, None, None)
    memo[target_N] = result
    return result

# Main execution part of the program
# Read the integer N from standard input
N = int(sys.stdin.readline())

# Handle the edge case N=1 explicitly. The only valid string is "1".
if N == 1:
    print("1")
    sys.exit()

# Initial Check: See if N itself can be represented as a string that meets all conditions.
is_valid_N_str, n_str = check_Q(N)
if is_valid_N_str:
    # If N forms a valid Q string, check the length constraint (1 to 1000).
    if 1 <= len(n_str) <= 1000:
        print(n_str)
        sys.exit()
    # If N is valid Q but too long, we must continue search for a factored form.
    # If N is not a valid Q (e.g., contains '0' or not palindrome), continue search.

# Filter the globally precomputed candidates list: only keep factors k <= N.
# This optimization avoids considering factors larger than N in the recursion.
current_candidates = [(k, p) for k, p in candidates if k <= N]

# Start the recursive search process for the input N
found_flag, P_list, q_str = solve(N)

final_solution_str = None # Initialize variable to store the final result string

if found_flag:
    # A valid decomposition structure (P_list, q_str) was found.
    # Construct the full palindrome string S based on this structure.
    m = len(P_list) # Number of P factors on one side
    if m == 0:
        # This scenario means solve(N) returned True directly from the base case check_Q(N).
        # This implies N itself forms the palindrome Q.
        # This path is usually covered by the initial check_Q(N) before calling solve().
        # Re-verify validity and length.
        is_valid_q, final_q_str = check_Q(N)
        if is_valid_q and 1 <= len(final_q_str) <= 1000:
             final_solution_str = final_q_str
        # If it fails length check here, final_solution_str remains None.
        
    else: # m > 0, P_list is not empty. Construct S = P1*...*Pm * Q * rev(Pm)*...*rev(P1)
        prefix_parts = P_list # List of P strings for the left side
        # Create suffix parts: first reverse each P string, then reverse the list order
        suffix_parts = [p[::-1] for p in P_list[::-1]] 
        
        # Join parts with '*'
        prefix_str = '*'.join(prefix_parts)
        suffix_str = '*'.join(suffix_parts)
        
        # Construct the final string S. The structure depends on whether Q represents value 1.
        if q_str == '1': # Middle part Q has value 1. Represents multiplicative identity.
             # String structure: P1 * ... * Pm * rev(Pm) * ... * rev(P1)
             # Requires exactly one '*' joining the prefix and suffix string blocks.
             S = prefix_str + '*' + suffix_str
        else: # Q is a non-'1' palindrome string. It must appear explicitly in the middle.
             # String structure: P1 * ... * Pm * Q * rev(Pm) * ... * rev(P1)
             # Requires '*' characters around the Q string part.
             S = prefix_str + '*' + q_str + '*' + suffix_str

        # Final check for the overall length constraint on the constructed string S
        if 1 <= len(S) <= 1000:
            final_solution_str = S
        # If S is too long (> 1000) or empty (unlikely), this constructed solution is invalid. 
        # final_solution_str remains None.

# Output the final result
if final_solution_str:
    # A valid string S satisfying all conditions was found and constructed.
    print(final_solution_str)
else:
    # No valid string S was found after exploring possibilities.
    print("-1")