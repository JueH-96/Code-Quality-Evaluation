import sys

def solve():
    T_str = sys.stdin.readline().strip()
    N_val = int(sys.stdin.readline())

    bags_data = []
    for _ in range(N_val):
        line_parts = sys.stdin.readline().split()
        # A_i = int(line_parts[0]) # Number of strings in bag, not strictly needed for parsing
        strings_in_bag = line_parts[1:] # The actual strings
        bags_data.append(strings_in_bag)

    len_T = len(T_str)
    
    # dp[j] will store the minimum cost to form the prefix T_str[0...j-1]
    # Maximum possible cost is N_val (if we pick one string from each bag).
    # So, N_val + 1 can serve as a representation for infinity.
    infinity = N_val + 1 
    
    # dp array for prefixes of T. dp[k] is min cost for T_str[0...k-1].
    # Size is len_T + 1 to store results for lengths from 0 up to len_T.
    dp = [infinity] * (len_T + 1)
    dp[0] = 0  # Cost to form an empty string (prefix of length 0) is 0.

    # Iterate through each bag one by one
    # Outer loop: i from 0 to N_val-1, representing current bag index
    for i in range(N_val): 
        current_bag_strings = bags_data[i]
        
        # next_dp will store the costs after considering bag `i`.
        # Initialize next_dp with current dp values. This handles the option:
        # "do nothing" with the current bag `i`.
        # The values in `dp` are results from considering bags 0 to `i-1`.
        next_dp = list(dp) # Shallow copy is sufficient as elements are integers
        
        # Iterate through all strings s_k available in the current bag `i`
        for s_k in current_bag_strings:
            len_sk = len(s_k)
            
            # Constraints state |S_i,j| is between 1 and 10.
            # So, len_sk will be >= 1. No need to check for len_sk == 0.

            # Try to form T_str[0...j-1] (a prefix of T of length j)
            # by appending s_k from the current bag.
            # If s_k is appended, it must match the suffix T_str[j-len_sk ... j-1].
            # The part before s_k, T_str[0...j-len_sk-1] (prefix of length j-len_sk),
            # must have been formed using bags 0 to `i-1`.
            # The cost for T_str[0...j-len_sk-1] is taken from `dp[j-len_sk]`.
            
            # Iterate j from len_sk to len_T.
            # j represents the length of the prefix of T we are trying to form.
            # Loop ensures j >= len_sk, so j-len_sk >= 0.
            for j in range(len_sk, len_T + 1):
                # Check if the suffix of T_str[0...j-1] (of length len_sk) matches s_k.
                # This suffix in Python string slicing is T_str[j-len_sk : j].
                if T_str[j-len_sk : j] == s_k:
                    # If T_str[0...j-len_sk-1] could be formed (i.e., cost is not infinity)
                    if dp[j-len_sk] != infinity:
                        # Cost to form T_str[0...j-1] is dp[j-len_sk] + 1 (cost of s_k).
                        # Update next_dp[j] if this path is cheaper.
                        next_dp[j] = min(next_dp[j], dp[j-len_sk] + 1)
        
        # After considering all strings in the current bag (bag `i`), 
        # and all ways to use them, `next_dp` contains the updated costs.
        # This becomes the `dp` table for the next iteration (for bag `i+1`).
        dp = next_dp

    result = dp[len_T] # Final minimum cost to form the entire string T.

    if result == infinity:
        sys.stdout.write("-1
")
    else:
        sys.stdout.write(str(result) + "
")

if __name__ == '__main__':
    solve()