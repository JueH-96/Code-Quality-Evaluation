mod = 998244353

import sys
import itertools

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    if H < W:
        new_grid = []
        for j in range(W):
            new_row = []
            for i in range(H):
                new_row.append(grid[i][j])
            new_grid.append(''.join(new_row))
        grid = new_grid
        H, W = W, H
    else:
        grid = [list(line) for line in grid]
    
    if W == 0:
        print(1)
        return

    states = list(itertools.product([0,1,2], repeat=W))
    n_state = len(states)
    base4_powers = [1] * W
    for j in range(1, W):
        base4_powers[j] = base4_powers[j-1] * 4
    size4 = base4_powers[W-1] * 4

    dp = [0] * n_state
    for state_id, state in enumerate(states):
        valid = True
        for j in range(1, W):
            if state[j] == state[j-1]:
                valid = False
                break
        if not valid:
            continue
        for j in range(W):
            if grid[0][j] != '?':
                digit_required = int(grid[0][j]) - 1
                if state[j] != digit_required:
                    valid = False
                    break
        if valid:
            dp[state_id] = 1

    if H == 1:
        total_ans = sum(dp) % mod
        print(total_ans)
        return

    n_subsets = 1 << W
    signs_list = [1] * n_subsets
    for s in range(n_subsets):
        bc = bin(s).count('1')
        if bc % 2 == 1:
            signs_list[s] = -1

    for i in range(1, H):
        P_arr = [0] * size4
        for state_id_prev in range(n_state):
            if dp[state_id_prev] == 0:
                continue
            state_prev = states[state_id_prev]
            idx4 = 0
            for j in range(W):
                d = state_prev[j]
                idx4 += d * base4_powers[j]
            P_arr[idx4] = (P_arr[idx4] + dp[state_id_prev]) % mod

        for j in range(W):
            for idx4 in range(size4-1, -1, -1):
                high = idx4 // base4_powers[j]
                digit = high % 4
                if digit == 3:
                    continue
                idx4_target = idx4 + (3 - digit) * base4_powers[j]
                P_arr[idx4_target] = (P_arr[idx4_target] + P_arr[idx4]) % mod

        base4_index0 = 0
        for j in range(W):
            base4_index0 += 3 * base4_powers[j]
        
        new_dp = [0] * n_state
        for state_id_curr in range(n_state):
            state_curr = states[state_id_curr]
            valid = True
            for j in range(1, W):
                if state_curr[j] == state_curr[j-1]:
                    valid = False
                    break
            if not valid:
                continue
            for j in range(W):
                if grid[i][j] != '?':
                    digit_required = int(grid[i][j]) - 1
                    if state_curr[j] != digit_required:
                        valid = False
                        break
            if not valid:
                continue

            total_ways = 0
            for s in range(n_subsets):
                base4_index = base4_index0
                for j in range(W):
                    if s & (1 << j):
                        base4_index += (state_curr[j] - 3) * base4_powers[j]
                total_ways = (total_ways + signs_list[s] * P_arr[base4_index]) % mod
            new_dp[state_id_curr] = total_ways % mod

        dp = new_dp

    total_ans = sum(dp) % mod
    print(total_ans)

if __name__ == '__main__':
    main()