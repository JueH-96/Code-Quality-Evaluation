MOD = 998244353

def main():
    import sys
    from collections import defaultdict

    S = sys.stdin.read().strip()
    dp = defaultdict(int)
    dp[('', '', False)] = 1  # initial state: no characters processed, no forbidden pattern

    for c in S:
        new_dp = defaultdict(int)
        for state in dp:
            last1, last2, forbidden = state
            count = dp[state]

            if c.isdigit():
                substitution_char = 'U'
            else:
                substitution_char = 'L'

            if substitution_char == 'U':
                new_last1 = last1
                new_last2 = last2
                new_last3 = ''

                if last1 == 'U' and last2 == 'U' and new_last3 == 'L':
                    new_state = (last1, last2, new_last3, True)
                    new_dp[new_state] = (new_dp[new_state] + count) % MOD
                else:
                    new_state = (new_last1, new_last2, new_last3, forbidden)
                    new_dp[new_state] = (new_dp[new_state] + count) % MOD
            elif substitution_char == 'L':
                new_last1 = 'U' if substitution_char == 'U' else 'L'
                new_last2 = last1
                new_last3 = last2

                if last1 == 'U' and last2 == 'U' and new_last3 == 'L':
                    new_state = (new_last1, new_last2, new_last3, True)
                    new_dp[new_state] = (new_dp[new_state] + count) % MOD
                else:
                    new_state = (new_last1, new_last2, new_last3, forbidden)
                    new_dp[new_state] = (new_dp[new_state] + count) % MOD
            else:
                if substitution_char == 'U':
                    new_last1 = 'U'
                    new_last2 = last1
                    new_last3 = last2

                    if last1 == 'U' and last2 == 'U' and new_last3 == 'L':
                        new_state = (new_last1, new_last2, new_last3, True)
                        new_dp[new_state] = (new_dp[new_state] + count) % MOD
                    else:
                        new_state = (new_last1, new_last2, new_last3, forbidden)
                        new_dp[new_state] = (new_dp[new_state] + count) % MOD
                else:
                    new_last1 = 'L'
                    new_last2 = last1
                    new_last3 = last2

                    if last1 == 'U' and last2 == 'U' and new_last3 == 'L':
                        new_state = (new_last1, new_last2, new_last3, True)
                        new_dp[new_state] = (new_dp[new_state] + count) % MOD
                    else:
                        new_state = (new_last1, new_last2, new_last3, forbidden)
                        new_dp[new_state] = (new_dp[new_state] + count) % MOD

        dp = new_dp

    ans = 0
    for state in dp:
        if not state[3]:
            ans = (ans + dp[state]) % MOD

    print(ans)

if __name__ == "__main__":
    main()