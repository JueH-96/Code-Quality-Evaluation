import sys

# Increase recursion limit for potentially deep recursions (e.g. for "(((...)))")
sys.setrecursionlimit(5000) 

MOD = 998244353

# Memoization table to store results for computed strings
memo = {}

# Precompute complements for parentheses
complement_map = {'(': ')', ')': '('}

def get_complement_reversed_str(s_str):
    """Computes the 'complement-reversed' version of s_str."""
    if not s_str:
        return ""
    s_len = len(s_str)
    # Pre-allocate list for characters for efficiency
    s_cr_list = [''] * s_len 
    for k in range(s_len):
        # s_str[s_len - 1 - k] is the k-th character from the end of s_str (0-indexed from end)
        # Its complement becomes the k-th character of s_cr_str (0-indexed from start)
        s_cr_list[k] = complement_map[s_str[s_len - 1 - k]]
    return "".join(s_cr_list)

def solve_recursive(s_str):
    """Recursively computes the number of distinct strings reachable from s_str."""
    if not s_str: # Base case: empty string
        return 1
    
    state = s_str # Use the string itself as the key for memoization
    if state in memo:
        return memo[state]

    s_len = len(s_str)
    
    # Determine if s_str is primitive. A non-empty VPS s is primitive if:
    # 1. s = (A) for some VPS A.
    # 2. s cannot be written as a concatenation S1S2 where S1, S2 are non-empty VPS.
    # This is equivalent to checking if balance of S[0...k] is > 0 for all k < |S|-1.
    is_primitive = True
    # All non-empty VPS start with '(' and end with ')'. This is guaranteed by problem statement.
    
    balance = 0
    # Iterate through S[0...k] for k from 0 to |S|-2.
    # If balance becomes 0 for any such prefix, S is not primitive.
    for k in range(s_len - 1): 
        if s_str[k] == '(':
            balance += 1
        else:
            balance -= 1
        if balance == 0: 
            is_primitive = False
            break
    
    if is_primitive:
        # Case 1: s_str = (A). Result is solve_recursive(A).
        # A is the content s_str[1...|S|-2]. Slice s_str[1:-1] gets this.
        A = s_str[1 : s_len-1] 
        res = solve_recursive(A)
        memo[state] = res
        return res
    else:
        # Case 2: s_str = P_1 P_2 ... P_m (concatenation of primitive VPSs P_i)
        # Each P_i is of the form (B_i). We need to calculate prod(solve_recursive(B_i)).
        prod_val = 1
        balance = 0 # Balance counter for decomposing s_str into P_i
        component_start_idx = 0
        for curr_idx in range(s_len):
            if s_str[curr_idx] == '(':
                balance += 1
            else:
                balance -= 1
            
            if balance == 0: 
                # Found a primitive component P_i = s_str[component_start_idx ... curr_idx]
                # B_i = P_i[1:-1] = s_str[component_start_idx+1 : curr_idx]
                B_i = s_str[component_start_idx+1 : curr_idx]
                
                prod_val = (prod_val * solve_recursive(B_i)) % MOD
                component_start_idx = curr_idx + 1
        
        # Compute s_str_CR, the complement-reversed version of s_str
        s_cr_str = get_complement_reversed_str(s_str)

        if s_str == s_cr_str:
            # If s_str is identical to its CR form, only one family of variations.
            memo[state] = prod_val
            return prod_val
        else:
            # Otherwise, two distinct families of variations (one from s_str, one from s_cr_str).
            res = (2 * prod_val) % MOD
            memo[state] = res
            return res

def main():
    # Read N (length of S), not strictly needed as len(S_initial) gives it.
    _ = int(sys.stdin.readline()) 
    S_initial = sys.stdin.readline().strip()
    
    ans = solve_recursive(S_initial)
    print(ans)

if __name__ == '__main__':
    main()