import sys
from collections import defaultdict

def main():
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Initialize DP
    dp = defaultdict(lambda: {'ways': 0, 'total': 0})
    dp[(0, 0, 0)] = {'ways': 1, 'total': 0}

    for char in S:
        new_dp = defaultdict(lambda: {'ways': 0, 'total': 0})
        for state in list(dp.keys()):
            a, b, c = state
            current_ways = dp[state]['ways']
            current_total = dp[state]['total']

            # Determine possible characters for this position
            if char == '?':
                possible_chars = ['A', 'B', 'C']
            else:
                possible_chars = [char]

            for ch in possible_chars:
                # Compute new state
                new_a = a ^ (1 if ch == 'A' else 0)
                new_b = b ^ (1 if ch == 'B' else 0)
                new_c = c ^ (1 if ch == 'C' else 0)

                # Calculate new_good
                new_good = 0
                for (a_prev, b_prev, c_prev) in dp:
                    if (new_a ^ a_prev) == (new_b ^ b_prev) and (new_b ^ b_prev) == (new_c ^ c_prev):
                        new_good += dp[(a_prev, b_prev, c_prev)]['ways']
                new_good %= MOD

                # Update new_dp
                new_dp[(new_a, new_b, new_c)]['ways'] = (new_dp[(new_a, new_b, new_c)]['ways'] + current_ways) % MOD
                new_dp[(new_a, new_b, new_c)]['total'] = (new_dp[(new_a, new_b, new_c)]['total'] + current_total + new_good * current_ways) % MOD

        dp = new_dp

    # Calculate the result
    result = 0
    for state in dp:
        if dp[state]['total'] >= K:
            result = (result + dp[state]['ways']) % MOD

    print(result % MOD)

if __name__ == '__main__':
    main()