# YOUR CODE HERE
import sys

def solve():
    # Read the input string S from standard input
    S = sys.stdin.readline().strip()
    # Get the length of the string S
    N = len(S)
    # Define the modulus for calculations
    MOD = 998244353

    # A valid parenthesis string must have an even length because the number of '('
    # must equal the number of ')'. If N is odd, it's impossible.
    if N % 2 != 0:
        print(0)
        return

    # Initialize the dynamic programming table (using space optimization with two rows implicitly)
    # dp_curr[j] will store the number of ways to form a prefix of the current length `i`
    # such that the balance (count '(' - count ')') is `j`, and the balance never went below 0.
    # The array size is N+1 to accommodate balances from 0 up to N.
    dp_curr = [0] * (N + 1) 
    
    # Base case: An empty prefix "" has length 0 and balance 0. There is 1 way to achieve this.
    dp_curr[0] = 1 

    # Iterate through each character of the input string S from index 0 to N-1
    # In each iteration `i`, we compute the DP states for prefixes of length `i+1` based on states for length `i`.
    for i in range(N):
        # Initialize the DP table for the next length (i+1) with zeros. This will store the results of this iteration.
        dp_next = [0] * (N + 1)
        
        # Determine the maximum relevant balance `j` for prefixes of the current length `i`.
        # The balance `j` must satisfy several conditions for a valid final string to be possible:
        # 1. `j >= 0`: Balance must always be non-negative. This condition is enforced by checks within the loop.
        # 2. `j <= i`: The maximum possible balance after `i` characters is `i` (if all characters were '(').
        # 3. `j <= N - i`: For a prefix of length `i` with balance `j`, it's only possible to eventually reach
        #    a final balance of 0 at length `N` if the current balance `j` is not more than the number 
        #    of remaining characters `N-i`. This is because each remaining character can decrease the balance by at most 1.
        # Combining conditions 2 and 3, the maximum relevant balance `j` we need to consider is `min(i, N - i)`.
        # Iterating `j` up to this limit prunes states that cannot possibly lead to a valid final string.
        
        limit_j = min(i, N - i) # This is the effective upper bound for j based on feasibility analysis

        # Iterate through possible balances `j` for prefixes of length `i`.
        # The range goes up to `limit_j` inclusive.
        for j in range(limit_j + 1): 
            # If there are no ways to reach the state (prefix length i, balance j), this state contributes nothing. Skip.
            if dp_curr[j] == 0:
                continue
            
            # Get the number of ways to reach the current state (i, j).
            val = dp_curr[j]
            # Get the character at the current position `i` in the input string S.
            char = S[i]

            # Consider the case where the character S[i] is fixed as '(' or is '?' and we choose to replace it with '('.
            # In this case, the balance increases by 1.
            if char == '(' or char == '?':
                # The new balance will be j+1. Check if this balance is within the maximum possible bound N.
                # Note: `j <= limit_j <= i`, so `j+1 <= i+1`. For `i < N`, `i+1 <= N`. This check `j+1 <= N` is generally true.
                if j + 1 <= N:
                    # Add the number of ways `val` to the count for the next state (prefix length i+1, balance j+1).
                    # Perform addition modulo MOD.
                    dp_next[j+1] = (dp_next[j+1] + val) % MOD
            
            # Consider the case where the character S[i] is fixed as ')' or is '?' and we choose to replace it with ')'.
            # In this case, the balance decreases by 1.
            if char == ')' or char == '?':
                # The new balance will be j-1. Check if this balance remains non-negative (>= 0).
                # This is a crucial condition for valid parenthesis strings: the balance must never drop below zero.
                if j - 1 >= 0: 
                    # Add the number of ways `val` to the count for the next state (prefix length i+1, balance j-1).
                    # Perform addition modulo MOD.
                    dp_next[j-1] = (dp_next[j-1] + val) % MOD
        
        # After calculating all contributions to `dp_next` based on `dp_curr`,
        # update `dp_curr` to `dp_next` to prepare for the next iteration (processing character S[i+1]).
        dp_curr = dp_next

    # After the loop finishes (all N characters processed), `dp_curr` holds the counts for strings of length N.
    # The final answer is the number of ways to form a complete string of length N that results in a final balance of 0.
    # This count is stored in `dp_curr[0]`.
    print(dp_curr[0])

# Execute the solver function to run the program logic.
solve()