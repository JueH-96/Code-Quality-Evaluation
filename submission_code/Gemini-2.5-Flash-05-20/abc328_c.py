import sys

def solve():
    # Read N and Q from the first line
    N, Q = map(int, sys.stdin.readline().split())
    # Read the string S from the second line
    S = sys.stdin.readline().strip()

    # Initialize the dp (prefix sum) array.
    # dp[i] will store the count of positions j such that S[j] == S[j+1]
    # for 0-indexed j where 0 <= j < i.
    # So, dp[k] effectively stores the sum of (1 if S[j]==S[j+1] else 0) for j from 0 to k-1.
    # The array size is N, as indices go from 0 to N-1.
    dp = [0] * N 

    # Build the prefix sum array
    # We iterate from i=1 to N-1 because dp[i] depends on S[i-1] and S[i].
    for i in range(1, N):
        # Inherit the count from the previous position
        dp[i] = dp[i-1]
        # Check if the characters at S[i-1] and S[i] are the same
        if S[i-1] == S[i]:
            # If they are, increment the count for this position
            dp[i] += 1

    # Process each query
    results = []
    for _ in range(Q):
        # Read l_i and r_i for the current query (1-indexed)
        l, r = map(int, sys.stdin.readline().split())

        # The problem asks for count of p such that l <= p <= r-1 and S_p = S_{p+1}.
        # Converting to 0-indexed: we need to count k such that (l-1) <= k <= (r-2) and S[k] = S[k+1].
        # Using the prefix sum array:
        # dp[r-1] contains the sum of matches for indices 0 to (r-2).
        # dp[l-1] contains the sum of matches for indices 0 to (l-2).
        # The difference dp[r-1] - dp[l-1] gives the sum of matches for indices (l-1) to (r-2).
        answer = dp[r-1] - dp[l-1]
        results.append(str(answer))
    
    # Print all results, each on a new line
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program
solve()