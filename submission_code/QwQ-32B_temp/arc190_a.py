import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    intervals = []
    for i in range(M):
        L, R = map(int, sys.stdin.readline().split())
        intervals.append((L, R))
    
    # Compute option 1: minimal 1-ops
    sorted_intervals_with_indices = sorted(enumerate(intervals), key=lambda x: x[1][1])
    selected_indices = []
    current_end = 0
    possible_A = False
    for idx, (L, R) in sorted_intervals_with_indices:
        if L > current_end:
            continue
        if R > current_end:
            selected_indices.append(idx)
            current_end = R
            if current_end >= N:
                possible_A = True
                break
    if current_end < N:
        possible_A = False
    else:
        possible_A = True

    # Compute option 2: minimal 2-ops (2 if possible)
    min_R = float('inf')
    min_R_idx = -1
    max_L = -float('inf')
    max_L_idx = -1
    for i in range(M):
        L, R = intervals[i]
        if R < min_R:
            min_R = R
            min_R_idx = i
        if L > max_L:
            max_L = L
            max_L_idx = i
    possible_B = (max_L > min_R)

    # Determine minimal cost
    cost_A = len(selected_indices) if possible_A else float('inf')
    cost_B = 2 if possible_B else float('inf')
    min_cost = min(cost_A, cost_B)
    
    if min_cost == float('inf'):
        print(-1)
        return
    
    # Choose the option with minimal cost
    if cost_A < cost_B:
        # Option 1: use selected indices as 1
        ops = [0] * M
        for idx in selected_indices:
            ops[idx] = 1
    elif cost_B < cost_A:
        # Option 2: use min_R and max_L intervals as 2
        ops = [0] * M
        ops[min_R_idx] = 2
        ops[max_L_idx] = 2
    else:
        # Tie, choose option 2 if possible
        if possible_B:
            ops = [0] * M
            ops[min_R_idx] = 2
            ops[max_L_idx] = 2
        else:
            ops = [0] * M
            for idx in selected_indices:
                ops[idx] = 1
    
    print(min_cost)
    print(' '.join(map(str, ops)))

if __name__ == "__main__":
    main()