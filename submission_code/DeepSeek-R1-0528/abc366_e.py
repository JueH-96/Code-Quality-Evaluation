import bisect

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    n = int(data[0])
    D_val = int(data[1])
    xs = []
    ys = []
    index = 2
    for i in range(n):
        x = int(data[index]); y = int(data[index+1]); index += 2
        xs.append(x)
        ys.append(y)
    
    if D_val == 0 and n > 0:
        pass
        
    xs.sort()
    ys.sort()
    
    P = [0] * (len(xs)+1)
    for i in range(1, len(P)):
        P[i] = P[i-1] + xs[i-1]
        
    Q = [0] * (len(ys)+1)
    for i in range(1, len(Q)):
        Q[i] = Q[i-1] + ys[i-1]
        
    def sum_x(x):
        pos = bisect.bisect_left(xs, x)
        left_sum = x * pos - P[pos]
        right_sum = (P[len(xs)] - P[pos]) - x * (len(xs) - pos)
        return left_sum + right_sum

    def sum_y(y):
        pos = bisect.bisect_left(ys, y)
        left_sum = y * pos - Q[pos]
        right_sum = (Q[len(ys)] - Q[pos]) - y * (len(ys) - pos)
        return left_sum + right_sum

    offset_x = (D_val // max(1, n)) if n > 0 else 0
    low_x_bound = xs[0] - offset_x - 1 if n > 0 else -1
    high_x_bound = xs[-1] + offset_x + 1 if n > 0 else 1

    L_arr = [None] * (D_val+1)
    R_arr = [None] * (D_val+1)
    
    for T_val in range(0, D_val+1):
        offset_y = T_val // max(1, n)
        low_y_bound = ys[0] - offset_y - 1 if n > 0 else -1
        high_y_bound = ys[-1] + offset_y + 1 if n > 0 else 1
        
        lo_y = low_y_bound
        hi_y = high_y_bound
        y_l_val = None
        while lo_y <= hi_y:
            mid_y = (lo_y + hi_y) // 2
            s_y_val = sum_y(mid_y) if n > 0 else 0
            if s_y_val <= T_val:
                y_l_val = mid_y
                hi_y = mid_y - 1
            else:
                lo_y = mid_y + 1
                
        if y_l_val is None:
            continue
            
        lo_y = low_y_bound
        hi_y = high_y_bound
        y_r_val = None
        while lo_y <= hi_y:
            mid_y = (lo_y + hi_y) // 2
            s_y_val = sum_y(mid_y) if n > 0 else 0
            if s_y_val <= T_val:
                y_r_val = mid_y
                lo_y = mid_y + 1
            else:
                hi_y = mid_y - 1
                
        L_arr[T_val] = y_l_val
        R_arr[T_val] = y_r_val
        
    total_count = 0
    if n == 0:
        if D_val >= 0:
            low_x_bound = - (D_val // 1) - 1
            high_x_bound = D_val // 1 + 1
            for x in range(low_x_bound, high_x_bound+1):
                for y in range(low_x_bound, high_x_bound+1):
                    total_count += 1
            print(total_count)
            return
        else:
            print(0)
            return

    for x in range(low_x_bound, high_x_bound+1):
        s_x_val = sum_x(x)
        if s_x_val > D_val:
            continue
        T_val = D_val - s_x_val
        if T_val < 0 or T_val > D_val:
            continue
        if L_arr[T_val] is None or R_arr[T_val] is None:
            continue
        total_count += (R_arr[T_val] - L_arr[T_val] + 1)
        
    print(total_count)

if __name__ == "__main__":
    main()