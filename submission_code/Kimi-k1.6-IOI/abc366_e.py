import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = int(input[ptr])
    ptr += 1
    
    x_list = []
    y_list = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr+1])
        x_list.append(x)
        y_list.append(y)
        ptr += 2
    
    # Process x coordinates
    sorted_x = sorted(x_list)
    prefix_x = [0] * (len(sorted_x) + 1)
    for i in range(len(sorted_x)):
        prefix_x[i+1] = prefix_x[i] + sorted_x[i]
    N_x = len(sorted_x)
    x_med_idx = N_x // 2
    x_med = sorted_x[x_med_idx]
    s_x_med = compute_sum(sorted_x, prefix_x, x_med)
    
    # Process y coordinates
    sorted_y = sorted(y_list)
    prefix_y = [0] * (len(sorted_y) + 1)
    for i in range(len(sorted_y)):
        prefix_y[i+1] = prefix_y[i] + sorted_y[i]
    N_y = len(sorted_y)
    y_med_idx = N_y // 2
    y_med = sorted_y[y_med_idx]
    s_y_med = compute_sum(sorted_y, prefix_y, y_med)
    
    if s_x_med > D or s_y_med > D:
        print(0)
        return
    
    # Find x_left and x_right using binary search
    def find_left(sorted_arr, prefix, D_val, med):
        low = -10**18
        high = med
        ans = med
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum(sorted_arr, prefix, mid)
            if s <= D_val:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans
    
    def find_right(sorted_arr, prefix, D_val, med):
        low = med
        high = 10**18
        ans = med
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum(sorted_arr, prefix, mid)
            if s <= D_val:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans
    
    x_left = find_left(sorted_x, prefix_x, D, x_med)
    x_right = find_right(sorted_x, prefix_x, D, x_med)
    
    # Generate all y in [y_left_initial, y_right_initial]
    y_left_initial = find_left(sorted_y, prefix_y, D, y_med)
    y_right_initial = find_right(sorted_y, prefix_y, D, y_med)
    
    sy_list = []
    for y in range(y_left_initial, y_right_initial + 1):
        s = compute_sum(sorted_y, prefix_y, y)
        sy_list.append(s)
    sy_list.sort()
    
    ans = 0
    for x in range(x_left, x_right + 1):
        s_x = compute_sum(sorted_x, prefix_x, x)
        T = D - s_x
        if T < 0:
            continue
        cnt = bisect.bisect_right(sy_list, T)
        ans += cnt
    
    print(ans)

def compute_sum(arr, prefix, x):
    k = bisect.bisect_right(arr, x)
    return x * (2 * k - len(arr)) + (prefix[-1] - 2 * prefix[k])

if __name__ == '__main__':
    main()