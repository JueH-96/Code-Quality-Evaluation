import sys

def main():
    N, D_thresh = map(int, sys.stdin.readline().split())
    points = []
    for _ in range(N):
        points.append(list(map(int, sys.stdin.readline().split())))

    MIN_COORD = -2 * 10**6 
    MAX_COORD = 2 * 10**6 
    RANGE_LEN = MAX_COORD - MIN_COORD + 1

    def compute_s_array(coords_list_orig, num_points_n):
        s_vals = [0] * RANGE_LEN
        
        coords_list = sorted(coords_list_orig)

        current_s_val = 0
        for c_val in coords_list:
            current_s_val += abs(MIN_COORD - c_val)
        s_vals[0] = current_s_val
        
        num_le_curr_x = 0 
        coord_ptr = 0
        
        # Initialize num_le_curr_x for current_x_val = MIN_COORD
        # These are points P_i such that P_i <= MIN_COORD
        current_x_val = MIN_COORD
        while coord_ptr < num_points_n and coords_list[coord_ptr] <= current_x_val:
            num_le_curr_x += 1
            coord_ptr += 1
            
        for i in range(RANGE_LEN - 1):
            # s_vals[i] stores S(MIN_COORD + i)
            # current_x_val is (MIN_COORD + i)
            # num_le_curr_x stores count of coords <= current_x_val
            
            slope = 2 * num_le_curr_x - num_points_n
            current_s_val += slope # This becomes S(MIN_COORD + i + 1)
            s_vals[i+1] = current_s_val
            
            # Update num_le_curr_x for next_x_val = MIN_COORD + i + 1
            current_x_val = MIN_COORD + i + 1 
            while coord_ptr < num_points_n and coords_list[coord_ptr] <= current_x_val:
                num_le_curr_x += 1
                coord_ptr += 1
        return s_vals

    x_all = [p[0] for p in points]
    y_all = [p[1] for p in points]

    arr_Sx = compute_s_array(x_all, N)
    arr_Sy = compute_s_array(y_all, N)

    y_all_sorted_temp = sorted(y_all) 
    y_median_val = y_all_sorted_temp[(N - 1) // 2] 
    
    y_median_idx = y_median_val - MIN_COORD
    y_median_idx = max(0, min(y_median_idx, RANGE_LEN - 1)) # Clamp to be safe

    total_count = 0

    for x_idx in range(RANGE_LEN):
        sx_val = arr_Sx[x_idx]
        
        Dy_thresh = D_thresh - sx_val
        
        if Dy_thresh < 0:
            continue

        low = 0
        high = y_median_idx 
        ans_low_yidx = -1
        while low <= high:
            mid = low + (high - low) // 2
            if arr_Sy[mid] <= Dy_thresh:
                ans_low_yidx = mid
                high = mid - 1 
            else:
                low = mid + 1
        
        low = y_median_idx 
        high = RANGE_LEN - 1
        ans_high_yidx = -1
        while low <= high:
            mid = low + (high - low) // 2
            if arr_Sy[mid] <= Dy_thresh:
                ans_high_yidx = mid
                low = mid + 1
            else:
                high = mid - 1
        
        if ans_low_yidx != -1 and ans_high_yidx != -1:
            total_count += (ans_high_yidx - ans_low_yidx + 1)
            
    print(total_count)

if __name__ == '__main__':
    main()