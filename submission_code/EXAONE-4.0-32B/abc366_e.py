import bisect

def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    n = int(data[0])
    D_val = int(data[1])
    points = []
    xs = []
    ys = []
    index = 2
    for i in range(n):
        x = int(data[index])
        y = int(data[index+1])
        index += 2
        xs.append(x)
        ys.append(y)
    
    if n == 0:
        if D_val >= 0:
            print("Infinity")
        else:
            print(0)
        return

    xs.sort()
    prefix_xs = [0] * n
    prefix_xs[0] = xs[0]
    for i in range(1, n):
        prefix_xs[i] = prefix_xs[i-1] + xs[i]
    total_sum_x = prefix_xs[-1]

    ys.sort()
    prefix_ys = [0] * n
    prefix_ys[0] = ys[0]
    for i in range(1, n):
        prefix_ys[i] = prefix_ys[i-1] + ys[i]
    total_sum_y = prefix_ys[-1]

    def f_x(x):
        if x <= xs[0]:
            return total_sum_x - n * x
        if x >= xs[-1]:
            return n * x - total_sum_x
        lo, hi = 0, n-1
        idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        left_count = idx + 1
        left_sum = prefix_xs[idx]
        right_count = n - left_count
        right_sum = total_sum_x - left_sum
        return (x * left_count - left_sum) + (right_sum - x * right_count)

    def g_y(y):
        if y <= ys[0]:
            return total_sum_y - n * y
        if y >= ys[-1]:
            return n * y - total_sum_y
        lo, hi = 0, n-1
        idx = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if ys[mid] <= y:
                idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        left_count = idx + 1
        left_sum = prefix_ys[idx]
        right_count = n - left_count
        right_sum = total_sum_y - left_sum
        return (y * left_count - left_sum) + (right_sum - y * right_count)

    low_x = xs[0] - (D_val // n) - 100
    high_x = xs[-1] + (D_val // n) + 100

    lo, hi = low_x, high_x
    left_bound_x = None
    while lo <= hi:
        mid = (lo + hi) // 2
        fm = f_x(mid)
        if fm <= D_val:
            left_bound_x = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if left_bound_x is None:
        print(0)
        return

    lo, hi = left_bound_x, high_x
    right_bound_x = left_bound_x
    while lo <= hi:
        mid = (lo + hi) // 2
        fm = f_x(mid)
        if fm <= D_val:
            right_bound_x = mid
            lo = mid + 1
        else:
            hi = mid - 1

    low_y = ys[0] - (D_val // n) - 100
    high_y = ys[-1] + (D_val // n) + 100

    lo, hi = low_y, high_y
    left_bound_y = None
    while lo <= hi:
        mid = (lo + hi) // 2
        gm = g_y(mid)
        if gm <= D_val:
            left_bound_y = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if left_bound_y is None:
        print(0)
        return

    lo, hi = left_bound_y, high_y
    right_bound_y = left_bound_y
    while lo <= hi:
        mid = (lo + hi) // 2
        gm = g_y(mid)
        if gm <= D_val:
            right_bound_y = mid
            lo = mid + 1
        else:
            hi = mid - 1

    arr_g = []
    for y_val in range(left_bound_y, right_bound_y + 1):
        arr_g.append(g_y(y_val))
    
    arr_g.sort()
    
    total_count = 0
    for x_val in range(left_bound_x, right_bound_x + 1):
        fx_val = f_x(x_val)
        remaining = D_val - fx_val
        if remaining < 0:
            continue
        pos = bisect.bisect_right(arr_g, remaining)
        total_count += pos
        
    print(total_count)

if __name__ == "__main__":
    main()