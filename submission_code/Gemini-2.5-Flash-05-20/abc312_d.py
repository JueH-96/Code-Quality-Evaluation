import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)
    MOD = 998244353

    # dp[j] will store the number of ways to form a prefix of current length (i)
    # such that its balance is j.
    # We use two arrays to optimize space: current_dp and next_dp.
    # current_dp represents dp[i] (for the prefix of length i)
    # next_dp represents dp[i+1] (for the prefix of length i+1)
    
    # Initialize current_dp for an empty prefix (length 0).
    # Balance is 0, there's 1 way (the empty string itself).
    # The array size is N+1 to accommodate balances from 0 to N.
    current_dp = [0] * (N + 1)
    current_dp[0] = 1

    # Iterate through each character in the input string S
    for i in range(N):
        # Initialize next_dp for the next length.
        # All counts for the next length start at 0 before processing transitions from current_dp.
        next_dp = [0] * (N + 1)
        char = S[i]

        # Iterate through all possible balance values for the current prefix length (i)
        for j in range(N + 1):
            # If there are no ways to reach this balance `j` at length `i`, skip.
            if current_dp[j] == 0:
                continue

            # Option 1: Current character S[i] becomes '(' (or it is already '(')
            # This is possible if S[i] is '(' or '?'.
            if char == '(' or char == '?':
                new_balance = j + 1
                # The balance must not exceed N (the maximum possible balance for a string of length N).
                # This check also ensures new_balance is within the bounds of our DP array (0 to N).
                if new_balance <= N: 
                    next_dp[new_balance] = (next_dp[new_balance] + current_dp[j]) % MOD

            # Option 2: Current character S[i] becomes ')' (or it is already ')')
            # This is possible if S[i] is ')' or '?'.
            if char == ')' or char == '?':
                new_balance = j - 1
                # The balance must not drop below 0 for a valid parenthesis string.
                if new_balance >= 0: 
                    next_dp[new_balance] = (next_dp[new_balance] + current_dp[j]) % MOD
        
        # After processing all possible transitions for S[i] from all possible `j` balances,
        # update current_dp for the next iteration (moving to process S[i+1]).
        current_dp = next_dp

    # After iterating through all characters of S (from S[0] to S[N-1]),
    # current_dp now holds the counts for strings of total length N.
    # The final answer is the number of ways to form a string of length N
    # that has a final balance of 0 and never dropped below 0.
    print(current_dp[0])

# Call the solve function to execute the program.
solve()