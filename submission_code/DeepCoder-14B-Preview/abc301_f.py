MOD = 998244353

def main():
    import sys
    from collections import defaultdict

    S = sys.stdin.readline().strip()

    uppercase = [chr(ord('A') + i) for i in range(26)]
    lowercase = [chr(ord('a') + i) for i in range(26)]
    all_chars = uppercase + lowercase

    # Initialize DP: state is a tuple (s, last_upper, third_lower)
    # where s is the state (0,1,2,3), last_upper is the last uppercase character (for s>=1),
    # third_lower is the third character (lowercase, for s=3)
    dp = defaultdict(int)
    dp[(0, None, None)] = 1

    for char in S:
        new_dp = defaultdict(int)
        if char == '?':
            possible_chars = all_chars
        else:
            possible_chars = [char]

        for state in dp:
            current_s, current_last_upper, current_third_lower = state
            current_count = dp[state]

            for c in possible_chars:
                if current_s == 0:
                    if c.isupper():
                        new_state = (1, c, None)
                        new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                    else:
                        new_dp[(0, None, None)] = (new_dp[(0, None, None)] + current_count) % MOD
                elif current_s == 1:
                    if c.isupper():
                        if c == current_last_upper:
                            new_state = (2, current_last_upper, None)
                            new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                        else:
                            new_state = (1, c, None)
                            new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                    else:
                        new_dp[(1, current_last_upper, None)] = (new_dp[(1, current_last_upper, None)] + current_count) % MOD
                elif current_s == 2:
                    if c.isupper():
                        if c == current_last_upper:
                            new_state = (2, current_last_upper, None)
                            new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                        else:
                            new_state = (1, c, None)
                            new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                    else:
                        new_state = (3, current_last_upper, c)
                        new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                elif current_s == 3:
                    if c.isupper():
                        if c == current_last_upper:
                            # This would complete the DDoS pattern, which is invalid
                            pass
                        else:
                            new_state = (1, c, None)
                            new_dp[new_state] = (new_dp[new_state] + current_count) % MOD
                    else:
                        # The state remains as s=3, third_lower remains the same
                        new_dp[(3, current_last_upper, current_third_lower)] = (new_dp[(3, current_last_upper, current_third_lower)] + current_count) % MOD

        dp = new_dp

    total = 0
    for state in dp:
        s, _, _ = state
        if s < 4:
            total = (total + dp[state]) % MOD
    print(total % MOD)

if __name__ == '__main__':
    main()