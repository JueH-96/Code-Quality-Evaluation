import math

def solve_problem():
    N = int(input())
    
    # Store points as dictionaries with original 0-indexed IDs
    # P_orig[i] corresponds to input P_{i+1}
    P_orig_list = []
    for i in range(N):
        a, b = map(int, input().split())
        P_orig_list.append({'x': a, 'y': b, 'id': i}) # id is original 0-based index for P
    
    Q_orig_list = []
    for i in range(N):
        c, d = map(int, input().split())
        Q_orig_list.append({'x': c, 'y': d, 'id': i}) # id is original 0-based index for Q

    # Create working lists of points that can be modified by popping
    # These lists will shrink in each iteration
    p_available = list(P_orig_list) 
    q_available = list(Q_orig_list)

    # ans[p_original_id] = q_original_id (both 0-indexed)
    # ans[i] will store the 0-indexed ID of Q point matched with P point of original 0-indexed ID i
    ans = [-1] * N 

    for _iter in range(N):
        # 1. Find p_pivot: point in p_available with min-X, then min-Y.
        # This point is guaranteed to exist as long as p_available is not empty.
        
        p_pivot_idx_in_list = 0 # Index of the pivot within the current p_available list
        for i in range(1, len(p_available)):
            p_curr = p_available[i]
            # p_candidate is the current best choice for p_pivot from p_available
            p_candidate = p_available[p_pivot_idx_in_list] 
            
            if p_curr['x'] < p_candidate['x']:
                p_pivot_idx_in_list = i
            elif p_curr['x'] == p_candidate['x']:
                if p_curr['y'] < p_candidate['y']:
                    p_pivot_idx_in_list = i
        
        p_pivot = p_available[p_pivot_idx_in_list]

        # 2. Find q_best: point in q_available that minimizes angle with p_pivot.
        # This point is guaranteed to exist as long as q_available is not empty.
        # Angle is measured CCW from positive x-axis. math.atan2(dy, dx) is in (-pi, pi].
        
        q_best_idx_in_list = 0 # Index of the best Q point within the current q_available list
        
        # Initialize min_angle with the angle for the first Q point in q_available
        # This handles the case where q_available has only one element.
        q_first_candidate = q_available[0]
        dx_first = q_first_candidate['x'] - p_pivot['x']
        dy_first = q_first_candidate['y'] - p_pivot['y']
        min_angle = math.atan2(dy_first, dx_first)

        # Iterate from the second Q point onwards (if any)
        for i in range(1, len(q_available)):
            q_curr = q_available[i]
            dx = q_curr['x'] - p_pivot['x']
            dy = q_curr['y'] - p_pivot['y']
            angle = math.atan2(dy, dx)
            
            if angle < min_angle:
                min_angle = angle
                q_best_idx_in_list = i
        
        q_best = q_available[q_best_idx_in_list]

        # 3. Record match using original 0-indexed IDs
        ans[p_pivot['id']] = q_best['id']

        # 4. Remove p_pivot and q_best from available lists
        # Popping by index ensures correct removal. These operations take O(length of list).
        p_available.pop(p_pivot_idx_in_list)
        q_available.pop(q_best_idx_in_list)
        
    # Prepare output: list of 1-indexed Q point IDs, in order of P point IDs
    # ans[i] is the 0-indexed Q ID for P with original 0-indexed ID i.
    # Output R_1, R_2, ..., R_N, where R_i is Q-index for P_i (1-indexed P, 1-indexed Q)
    # So ans[0] (for P_1) should be Q_id+1, ans[1] (for P_2) should be Q_id+1 etc.
    output_values = [val + 1 for val in ans]
    print(*(output_values))

# Call the main solving function
solve_problem()