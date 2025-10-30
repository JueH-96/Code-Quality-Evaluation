def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    S = input_data[2]

    MOD = 998244353

    # We use dynamic programming in which dp[i] is a dictionary:
    #   key   = suffix (a string of length < K, holding the last characters added)
    #   value = number of ways (modulo MOD) to form a valid string up to index i-1
    #
    # At step i, we consider S[i], which can be 'A', 'B', or '?'. 
    # If it is '?', we branch to both 'A' and 'B'. 
    # Otherwise, we only have one choice.
    #
    # After appending the chosen character to the suffix from dp[i], we check if
    # the last K characters form a palindrome. If they do, we skip it; if they don't,
    # we keep the new suffix of length up to K-1.
    #
    # Finally, dp[N] holds the counts of all valid strings that do not contain a
    # palindromic substring of length K anywhere.

    from collections import defaultdict

    # dp[i]: dictionary mapping "suffix string" -> ways
    dp = [defaultdict(int) for _ in range(N+1)]
    dp[0][""] = 1  # No characters chosen yet, 1 way.

    for i in range(N):
        next_dp = defaultdict(int)
        ch = S[i]

        # For each possible suffix in dp[i]
        for suffix, ways in dp[i].items():
            # Determine what characters we can place here
            if ch == '?':
                possible_chars = ['A', 'B']
            else:
                possible_chars = [ch]

            for c in possible_chars:
                new_suffix = suffix + c
                # Check if the last K characters form a palindrome
                if len(new_suffix) >= K:
                    lastK = new_suffix[-K:]
                    if lastK == lastK[::-1]:
                        # This would form a palindrome of length K, skip
                        continue

                # Keep only the last K-1 characters as the suffix for future checks
                if len(new_suffix) >= K-1:
                    final_suffix = new_suffix[-(K-1):]
                else:
                    final_suffix = new_suffix

                next_dp[final_suffix] = (next_dp[final_suffix] + ways) % MOD

        dp[i+1] = next_dp

    # Sum up all ways in dp[N]
    answer = sum(dp[N].values()) % MOD
    print(answer)