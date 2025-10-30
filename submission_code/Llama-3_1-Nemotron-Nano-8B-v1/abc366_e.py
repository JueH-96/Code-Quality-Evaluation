import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    D = int(input[idx])
    idx +=1
    x = []
    y = []
    for _ in range(N):
        xi = int(input[idx])
        yi = int(input[idx+1])
        x.append(xi)
        y.append(yi)
        idx +=2
    
    # Sort x and compute prefix sum
    sorted_x = sorted(x)
    prefix_x = [0]*(N+1)
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + sorted_x[i]
    
    # Sort y and compute prefix sum
    sorted_y = sorted(y)
    prefix_y = [0]*(N+1)
    for i in range(N):
        prefix_y[i+1] = prefix_y[i] + sorted_y[i]
    
    # Compute median_x
    if N %2 ==1:
        median_x = sorted_x[N//2]
    else:
        median_x = (sorted_x[N//2 -1] + sorted_x[N//2])/2.0
    floor_median_x = int(median_x)
    if median_x != floor_median_x:
        floor_median_x +=1 if median_x > floor_median_x else 0
    ceil_median_x = int(median_x) +1 if median_x != floor_median_x else floor_median_x
    
    # Compute S_min_x
    def compute_sum_x(x_val):
        if not sorted_x:
            return 0
        pos = bisect.bisect_left(sorted_x, x_val)
        left = pos
        right = N - pos
        sum_left = x_val * left - prefix_x[pos]
        sum_right = (prefix_x[N] - prefix_x[pos]) - x_val * right
        return sum_left + sum_right
    
    if N %2 ==1:
        s_min_x = compute_sum_x(median_x)
    else:
        s_min_x = compute_sum_x(sorted_x[N//2 -1])
    
    # Compute S_min_y
    def compute_sum_y(y_val):
        if not sorted_y:
            return 0
        pos = bisect.bisect_left(sorted_y, y_val)
        left = pos
        right = N - pos
        sum_left = y_val * left - prefix_y[pos]
        sum_right = (prefix_y[N] - prefix_y[pos]) - y_val * right
        return sum_left + sum_right
    
    if N %2 ==1:
        s_min_y = compute_sum_y(median_y)
    else:
        s_min_y = compute_sum_y(sorted_y[N//2 -1])
    median_y = (sorted_y[N//2 -1] + sorted_y[N//2])/2.0 if N%2 ==0 else sorted_y[N//2]
    
    if s_min_x + s_min_y > D:
        print(0)
        return
    
    # Find left_part_max: largest x <= floor_median_x with sum_x <= D
    def find_left_part_max():
        low = -10**18
        high = floor_median_x
        best = -10**18
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum_x(mid)
            if s <= D:
                best = mid
                low = mid +1
            else:
                high = mid -1
        return best
    
    left_part_max = find_left_part_max()
    
    # Find right_part_min: largest x >= ceil_median_x with sum_x <= D
    def find_right_part_min():
        low = ceil_median_x
        high = 10**18
        best = 10**18
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum_x(mid)
            if s <= D:
                best = mid
                low = mid +1
            else:
                high = mid -1
        return best
    
    right_part_min = find_right_part_min()
    
    if left_part_max > right_part_min:
        print(0)
        return
    
    # Compute count for each x in [left_part_max, right_part_min]
    total =0
    for x_val in range(left_part_max, right_part_min +1):
        s_x = compute_sum_x(x_val)
        rem = D - s_x
        if rem <0:
            continue
        
        # Compute count_y where sum_y(y) <= rem
        if rem < s_min_y:
            continue
        
        # Find lower_y and upper_y
        def find_lower_y(rem_y):
            low = -10**18
            high = (sorted_y[N//2 -1] + sorted_y[N//2])//2 if N%2 ==0 else sorted_y[N//2]
            best = None
            while low <= high:
                mid = (low + high) //2
                s = compute_sum_y(mid)
                if s <= rem_y:
                    best = mid
                    high = mid -1
                else:
                    low = mid +1
            return best
        
        def find_upper_y(rem_y):
            low = (sorted_y[N//2 -1] + sorted_y[N//2])//2 if N%2 ==0 else sorted_y[N//2]
            high = 10**18
            best = None
            while low <= high:
                mid = (low + high) //2
                s = compute_sum_y(mid)
                if s <= rem_y:
                    best = mid
                    low = mid +1
                else:
                    high = mid -1
            return best
        
        lower = find_lower_y(rem)
        upper = find_upper_y(rem)
        if lower is None or upper is None or upper < lower:
            continue
        total += upper - lower +1
    
    print(total)

if __name__ == '__main__':
    main()