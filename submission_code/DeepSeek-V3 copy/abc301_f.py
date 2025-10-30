MOD = 998244353

def count_valid_strings(S):
    n = len(S)
    if n < 4:
        return 0
    # Precompute the number of ?s and their positions
    q_indices = [i for i, c in enumerate(S) if c == '?']
    q = len(q_indices)
    # Precompute the number of ways to assign letters to ?s
    # For each ?, it can be any of 52 letters (26 uppercase + 26 lowercase)
    # So total ways is 52^q
    total_ways = pow(52, q, MOD)
    # Now, we need to subtract the number of strings that contain a DDoS-type subsequence
    # A DDoS-type subsequence is of the form A A a A, where A is an uppercase letter, a is a lowercase letter
    # We need to count the number of strings that have at least one such subsequence
    # To do this, we can use inclusion-exclusion principle
    # But it's easier to use dynamic programming to count the number of strings that do not contain any DDoS-type subsequence
    # We will use a DP approach to count the number of valid strings
    # The DP state will keep track of the current position and the last few characters that could form a DDoS-type subsequence
    # We will use a state that represents the last three characters and whether they could form a DDoS-type subsequence
    # Initialize DP
    # dp[i][state] where state represents the last three characters and whether they could form a DDoS-type subsequence
    # Since the state can be complex, we will use a more efficient approach
    # Instead of tracking all possible states, we will track the number of ways to form valid strings up to the current position
    # We will use a DP table where dp[i][j] represents the number of ways to form a valid string up to position i, with j representing the state
    # The state j can be:
    # 0: no characters have been selected yet
    # 1: one uppercase character has been selected
    # 2: two uppercase characters have been selected
    # 3: two uppercase characters and one lowercase character have been selected
    # 4: three uppercase characters have been selected
    # We will initialize dp[0][0] = 1
    dp = [ [0] * 5 for _ in range(n+1) ]
    dp[0][0] = 1
    for i in range(n):
        c = S[i]
        for j in range(5):
            if dp[i][j] == 0:
                continue
            if c == '?':
                # Assign any uppercase letter
                # Update state based on current state
                if j == 0:
                    dp[i+1][1] = (dp[i+1][1] + dp[i][j] * 26) % MOD
                elif j == 1:
                    dp[i+1][2] = (dp[i+1][2] + dp[i][j] * 26) % MOD
                elif j == 2:
                    dp[i+1][4] = (dp[i+1][4] + dp[i][j] * 26) % MOD
                elif j == 3:
                    dp[i+1][4] = (dp[i+1][4] + dp[i][j] * 26) % MOD
                elif j == 4:
                    dp[i+1][4] = (dp[i+1][4] + dp[i][j] * 26) % MOD
                # Assign any lowercase letter
                if j == 2:
                    dp[i+1][3] = (dp[i+1][3] + dp[i][j] * 26) % MOD
                else:
                    dp[i+1][j] = (dp[i+1][j] + dp[i][j] * 26) % MOD
            else:
                if c.isupper():
                    if j == 0:
                        dp[i+1][1] = (dp[i+1][1] + dp[i][j]) % MOD
                    elif j == 1:
                        dp[i+1][2] = (dp[i+1][2] + dp[i][j]) % MOD
                    elif j == 2:
                        dp[i+1][4] = (dp[i+1][4] + dp[i][j]) % MOD
                    elif j == 3:
                        dp[i+1][4] = (dp[i+1][4] + dp[i][j]) % MOD
                    elif j == 4:
                        dp[i+1][4] = (dp[i+1][4] + dp[i][j]) % MOD
                else:
                    if j == 2:
                        dp[i+1][3] = (dp[i+1][3] + dp[i][j]) % MOD
                    else:
                        dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD
    # The total number of valid strings is the sum of dp[n][j] for j in 0..4
    valid_ways = sum(dp[n][j] for j in range(5)) % MOD
    return valid_ways

S = input().strip()
print(count_valid_strings(S))