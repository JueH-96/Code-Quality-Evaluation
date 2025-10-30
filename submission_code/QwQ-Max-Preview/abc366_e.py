import bisect

def compute_Sx(x, sorted_x, prefix_sum_x):
    n = len(sorted_x)
    k = bisect.bisect_right(sorted_x, x)
    sum_left = x * k - prefix_sum_x[k]
    sum_right = (prefix_sum_x[-1] - prefix_sum_x[k]) - x * (n - k)
    return sum_left + sum_right

def compute_Sy(y, sorted_y, prefix_sum_y):
    n = len(sorted_y)
    k = bisect.bisect_right(sorted_y, y)
    sum_left = y * k - prefix_sum_y[k]
    sum_right = (prefix_sum_y[-1] - prefix_sum_y[k]) - y * (n - k)
    return sum_left + sum_right

def find_x_left(sorted_x, prefix_sum_x, D, median_x, s_min_x):
    if s_min_x > D:
        return None
    low = -10**18
    high = median_x
    x_left = median_x
    while low <= high:
        mid = (low + high) // 2
        s = compute_Sx(mid, sorted_x, prefix_sum_x)
        if s <= D:
            x_left = mid
            high = mid - 1
        else:
            low = mid + 1
    return x_left

def find_x_right(sorted_x, prefix_sum_x, D, median_x, s_min_x):
    if s_min_x > D:
        return None
    low = median_x
    high = 10**18
    x_right = median_x
    while low <= high:
        mid = (low + high) // 2
        s = compute_Sx(mid, sorted_x, prefix_sum_x)
        if s <= D:
            x_right = mid
            low = mid + 1
        else:
            high = mid - 1
    return x_right

def find_y_left(sorted_y, prefix_sum_y, s_y_max, median_y, s_min_y):
    if s_min_y > s_y_max:
        return None
    low = -10**18
    high = median_y
    y_left = median_y
    while low <= high:
        mid = (low + high) // 2
        s = compute_Sy(mid, sorted_y, prefix_sum_y)
        if s <= s_y_max:
            y_left = mid
            high = mid - 1
        else:
            low = mid + 1
    return y_left

def find_y_right(sorted_y, prefix_sum_y, s_y_max, median_y, s_min_y):
    if s_min_y > s_y_max:
        return None
    low = median_y
    high = 10**18
    y_right = median_y
    while low <= high:
        mid = (low + high) // 2
        s = compute_Sy(mid, sorted_y, prefix_sum_y)
        if s <= s_y_max:
            y_right = mid
            low = mid + 1
        else:
            high = mid - 1
    return y_right

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = int(input[ptr])
    ptr += 1
    x_coords = []
    y_coords = []
    for _ in range(N):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        x_coords.append(x)
        y_coords.append(y)
    
    # Process x coordinates
    sorted_x = sorted(x_coords)
    prefix_sum_x = [0] * (N + 1)
    for i in range(N):
        prefix_sum_x[i + 1] = prefix_sum_x[i] + sorted_x[i]
    median_x = sorted_x[N // 2]
    s_min_x = compute_Sx(median_x, sorted_x, prefix_sum_x)
    if s_min_x > D:
        print(0)
        return
    
    x_left = find_x_left(sorted_x, prefix_sum_x, D, median_x, s_min_x)
    x_right = find_x_right(sorted_x, prefix_sum_x, D, median_x, s_min_x)
    if x_left is None or x_right is None:
        print(0)
        return
    x_count = x_right - x_left + 1
    if x_count <= 0:
        print(0)
        return
    
    # Process y coordinates
    sorted_y = sorted(y_coords)
    prefix_sum_y = [0] * (N + 1)
    for i in range(N):
        prefix_sum_y[i + 1] = prefix_sum_y[i] + sorted_y[i]
    median_y = sorted_y[N // 2]
    s_min_y = compute_Sy(median_y, sorted_y, prefix_sum_y)
    
    # Iterate over x in x_left to x_right
    total = 0
    current_x = x_left
    current_s_x = compute_Sx(current_x, sorted_x, prefix_sum_x)
    for delta in range(x_right - x_left + 1):
        x = current_x + delta
        if delta == 0:
            s_x = current_s_x
        else:
            previous_x = x - 1
            k_prev = bisect.bisect_right(sorted_x, previous_x)
            s_x = current_s_x + (2 * k_prev - N)
            current_s_x = s_x
        
        s_y_max = D - s_x
        if s_y_max < s_min_y:
            continue
        
        yl = find_y_left(sorted_y, prefix_sum_y, s_y_max, median_y, s_min_y)
        yr = find_y_right(sorted_y, prefix_sum_y, s_y_max, median_y, s_min_y)
        if yl is None or yr is None:
            continue
        total += (yr - yl + 1)
    
    print(total)

if __name__ == '__main__':
    main()