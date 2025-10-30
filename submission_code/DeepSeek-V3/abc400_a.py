import sys

def solve():
    N, R, C = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    prefix_r = [0] * (N + 1)
    prefix_c = [0] * (N + 1)
    
    # Compute prefix sums for r and c
    for i in range(1, N+1):
        direction = S[i-1]
        dr = 0
        dc = 0
        if direction == 'N':
            dr = -1
        elif direction == 'S':
            dr = 1
        elif direction == 'W':
            dc = -1
        elif direction == 'E':
            dc = 1
        prefix_r[i] = prefix_r[i-1] + dr
        prefix_c[i] = prefix_c[i-1] + dc
    
    # We need to find for each t, if there exists k <=t such that:
    # prefix_r[k-1] = prefix_r[t] + R
    # prefix_c[k-1] = prefix_c[t] + C
    
    # So for each t, the target is (pr_t + R, pc_t + C), and we need to check if this pair exists in the prefix up to t-1.
    
    # We'll use a dictionary to map (pr, pc) to the earliest occurrence.
    coord_map = {}
    coord_map[(0, 0)] = 0  # prefix_r[0] and prefix_c[0]
    
    res = []
    for t in range(1, N+1):
        target_r = prefix_r[t] + R
        target_c = prefix_c[t] + C
        # Check if (target_r, target_c) exists in the prefix up to t-1 (i.e., 0..t-1)
        if (target_r, target_c) in coord_map:
            res.append('1')
        else:
            res.append('0')
        # Add current prefix_r[t] and prefix_c[t] to the map if not present
        current_r = prefix_r[t]
        current_c = prefix_c[t]
        if (current_r, current_c) not in coord_map:
            coord_map[(current_r, current_c)] = t
    print(''.join(res))

solve()