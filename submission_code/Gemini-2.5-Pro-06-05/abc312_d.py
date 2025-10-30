import sys

def main():
    """
    Reads the input string, solves the problem using dynamic programming,
    and prints the result.
    """
    S = sys.stdin.readline().strip()
    N = len(S)
    MOD = 998244353

    # A valid parenthesis string must have an even length.
    if N % 2 != 0:
        print(0)
        return

    # dp[j]: number of ways to form a prefix of the current length
    # with a balance of j (number of open brackets - number of closed brackets).
    
    # Initialize for a prefix of length 0 (the empty string).
    # There is 1 way to have a balance of 0.
    dp = [0] * (N + 1)
    dp[0] = 1

    # Process each character of the string S one by one.
    for i in range(N):
        new_dp = [0] * (N + 1)
        char = S[i]

        # Iterate over possible balances `j` for the prefix of length `i`.
        # The balance `j` can be at most `i`.
        for j in range(i + 1):
            if dp[j] == 0:
                continue
            
            # Case 1: Place a closing parenthesis ')'.
            # This is possible if the character is ')' or '?'.
            # The balance decreases, so it must be positive before this move
            # to avoid becoming negative.
            if char in (')', '?'):
                if j > 0:
                    new_dp[j - 1] = (new_dp[j - 1] + dp[j]) % MOD

            # Case 2: Place an opening parenthesis '('.
            # This is possible if the character is '(' or '?'.
            # The balance increases.
            if char in ('(', '?'):
                # The new balance `j+1` must not exceed N.
                if j + 1 <= N:
                    new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % MOD
        
        # The new dp state becomes the current state for the next iteration.
        dp = new_dp

    # The final answer is the number of ways for the full string of length N
    # to have a balance of 0.
    print(dp[0])

if __name__ == "__main__":
    main()