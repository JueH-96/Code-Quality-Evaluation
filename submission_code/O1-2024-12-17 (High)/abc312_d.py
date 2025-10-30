def main():
    import sys
    MOD = 998244353
    
    S = sys.stdin.readline().strip()
    n = len(S)

    # dp[j] will store the number of ways to parse the first i characters
    # (where i is the loop index) with j "unmatched" '(' brackets.
    dp = [0] * (n + 1)
    dp[0] = 1  # There's exactly 1 way to have 0 unmatched '(' before processing any character

    for i, ch in enumerate(S):
        new_dp = [0] * (n + 1)
        if ch == '(':
            # Only option is to open an additional bracket
            for j in range(i + 1):
                if dp[j]:
                    val = dp[j]
                    new_dp[j + 1] = (new_dp[j + 1] + val) % MOD
        elif ch == ')':
            # Only option is to close a bracket
            for j in range(1, i + 2):
                if dp[j]:
                    val = dp[j]
                    new_dp[j - 1] = (new_dp[j - 1] + val) % MOD
        else:  # ch == '?', can act as '(' or ')'
            for j in range(i + 1):
                if dp[j]:
                    val = dp[j]
                    # Act as '('
                    new_dp[j + 1] = (new_dp[j + 1] + val) % MOD
                    # Act as ')'
                    if j > 0:
                        new_dp[j - 1] = (new_dp[j - 1] + val) % MOD

        dp = new_dp

    print(dp[0] % MOD)

# Don't forget to call main()!
main()