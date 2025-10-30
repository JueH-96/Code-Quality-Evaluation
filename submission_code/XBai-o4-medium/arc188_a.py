import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    S = input[idx]
    
    # Initialize DP and cumulative_gsc
    previous_dp = defaultdict(lambda: defaultdict(int))
    initial_state = (0, 0, 0)
    previous_dp[initial_state][0] = 1
    previous_cumulative_gsc = defaultdict(int)
    previous_cumulative_gsc[initial_state] = 1

    for i in range(N):
        current_char = S[i]
        current_dp = defaultdict(lambda: defaultdict(int))
        current_gsc_step_i = defaultdict(int)
        for state_prev in previous_dp:
            for s_prev in previous_dp[state_prev]:
                count_ways = previous_dp[state_prev][s_prev]
                x_prev, y_prev, z_prev = state_prev
                if current_char == '?':
                    for c in ['A', 'B', 'C']:
                        if c == 'A':
                            x_new = (x_prev + 1) % 2
                            y_new = y_prev
                            z_new = z_prev
                        elif c == 'B':
                            x_new = x_prev
                            y_new = (y_prev + 1) % 2
                            z_new = z_prev
                        else:  # 'C'
                            x_new = x_prev
                            y_new = y_prev
                            z_new = (z_prev + 1) % 2
                        new_state = (x_new, y_new, z_new)
                        comp = (1 - x_new, 1 - y_new, 1 - z_new)
                        add_sub = previous_cumulative_gsc.get(new_state, 0) + previous_cumulative_gsc.get(comp, 0)
                        new_s = s_prev + add_sub
                        current_dp[new_state][new_s] = (current_dp[new_state][new_s] + count_ways) % MOD
                        current_gsc_step_i[new_state] = (current_gsc_step_i[new_state] + count_ways) % MOD
                else:
                    if current_char == 'A':
                        x_new = (x_prev + 1) % 2
                        y_new = y_prev
                        z_new = z_prev
                    elif current_char == 'B':
                        x_new = x_prev
                        y_new = (y_prev + 1) % 2
                        z_new = z_prev
                    else:  # 'C'
                        x_new = x_prev
                        y_new = y_prev
                        z_new = (z_prev + 1) % 2
                    new_state = (x_new, y_new, z_new)
                    comp = (1 - x_new, 1 - y_new, 1 - z_new)
                    add_sub = previous_cumulative_gsc.get(new_state, 0) + previous_cumulative_gsc.get(comp, 0)
                    new_s = s_prev + add_sub
                    current_dp[new_state][new_s] = (current_dp[new_state][new_s] + count_ways) % MOD
                    current_gsc_step_i[new_state] = (current_gsc_step_i[new_state] + count_ways) % MOD
        # Update cumulative_gsc
        new_cumulative_gsc = previous_cumulative_gsc.copy()
        for state in current_gsc_step_i:
            new_cumulative_gsc[state] = (new_cumulative_gsc.get(state, 0) + current_gsc_step_i[state]) % MOD
        # Prepare for next iteration
        previous_dp = current_dp
        previous_cumulative_gsc = new_cumulative_gsc

    # Calculate the result
    result = 0
    for state in previous_dp:
        for s in previous_dp[state]:
            if s >= K:
                result = (result + previous_dp[state][s]) % MOD
    print(result % MOD)

if __name__ == '__main__':
    main()