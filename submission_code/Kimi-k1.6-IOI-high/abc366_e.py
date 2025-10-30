import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    D = int(data[1])
    
    points = list(map(int, data[2:]))
    xs = sorted(points[0::2])
    ys = sorted(points[1::2])
    
    if n == 0:
        print(0)
        return
    
    # Compute prefix sums
    prefix_x = [0] * (n + 1)
    for i in range(n):
        prefix_x[i+1] = prefix_x[i] + xs[i]
    
    prefix_y = [0] * (n + 1)
    for i in range(n):
        prefix_y[i+1] = prefix_y[i] + ys[i]
    
    def sumx(x):
        k = bisect.bisect_right(xs, x)
        return x * k - prefix_x[k] + (prefix_x[-1] - prefix_x[k]) - x * (n - k)
    
    def sumy(y):
        k = bisect.bisect_right(ys, y)
        return y * k - prefix_y[k] + (prefix_y[-1] - prefix_y[k]) - y * (n - k)
    
    x_med = xs[n//2] if n > 0 else 0
    y_med = ys[n//2] if n > 0 else 0
    
    sumx_med = sumx(x_med) if n > 0 else 0
    sumy_med = sumy(y_med) if n > 0 else 0
    
    if sumx_med + sumy_med > D:
        print(0)
        return
    
    sum_xi = prefix_x[-1] if n > 0 else 0
    sum_x_sorted0 = sum_xi - xs[0] * n if n > 0 else 0
    sum_x_sorted_last = sum_xi - xs[-1] * n if n > 0 else 0
    
    steps_left_x = 0
    if n > 0 and sum_x_sorted0 <= D:
        steps_left_x = (D - sum_x_sorted0) // n
    
    count1_x = max(0, steps_left_x) if n > 0 else 0
    region1_x_start = xs[0] - steps_left_x if n > 0 else 0
    region1_x_end = xs[0] - 1 if n > 0 else 0
    
    steps_right_x = 0
    if n > 0 and sum_x_sorted_last <= D:
        steps_right_x = (D - sum_x_sorted_last) // n
    
    count3_x = max(0, steps_right_x) if n > 0 else 0
    region3_x_start = xs[-1] + 1 if n > 0 else 0
    region3_x_end = xs[-1] + steps_right_x if n > 0 else 0
    
    x_low = x_med
    x_high = x_med
    
    if n > 0:
        left = xs[0]
        right = x_med
        while left <= right:
            mid = (left + right) // 2
            s = sumx(mid)
            if s <= D:
                x_low = mid
                right = mid - 1
            else:
                left = mid + 1
        
        left = x_med
        right = xs[-1]
        while left <= right:
            mid = (left + right) // 2
            s = sumx(mid)
            if s <= D:
                x_high = mid
                left = mid + 1
            else:
                right = mid - 1
    
    count2_x = max(0, x_high - x_low + 1) if n > 0 else 0
    
    total = 0
    
    def count_valid_y(rem):
        nonlocal ys, prefix_y, n, sumy_med
        if n == 0 or rem < sumy_med:
            return 0
        
        sumyi = prefix_y[-1]
        sumy_sorted0 = sumyi - ys[0] * n if n > 0 else 0
        sumy_sorted_last = sumyi - ys[-1] * n if n > 0 else 0
        
        region1_count = 0
        if n > 0 and sumy_sorted0 <= rem:
            numerator = sumyi - rem
            lower_bound = (numerator + n - 1) // n
            region1_count = max(ys[0] - lower_bound, 0) if n > 0 else 0
        
        region3_count = 0
        if n > 0 and sumy_sorted_last <= rem:
            numerator = sumyi + rem
            y_max = numerator // n
            region3_count = max(y_max - ys[-1], 0) if n > 0 else 0
        
        if n == 0:
            return region1_count + region3_count
        
        y_med_val = ys[n//2] if n > 0 else 0
        
        # Find y_low
        y_low = y_med_val
        left = ys[0]
        right = y_med_val
        while left <= right:
            mid = (left + right) // 2
            s = sumy(mid)
            if s <= rem:
                y_low = mid
                right = mid - 1
            else:
                left = mid + 1
        
        # Find y_high
        y_high = y_med_val
        left = y_med_val
        right_val = ys[-1] if n > 0 else 0
        while left <= right_val:
            mid = (left + right_val) // 2
            s = sumy(mid)
            if s <= rem:
                y_high = mid
                left = mid + 1
            else:
                right_val = mid - 1
        
        region2_count = max(0, y_high - y_low + 1) if n > 0 else 0
        return region1_count + region2_count + region3_count
    
    # Iterate through all valid x regions
    if n > 0 and region1_x_start <= region1_x_end:
        for x in range(region1_x_start, region1_x_end + 1):
            sumx_x = sumx(x)
            rem = D - sumx_x
            if rem >= 0 and rem >= sumy_med:
                total += count_valid_y(rem)
    
    if n > 0 and x_low <= x_high:
        for x in range(x_low, x_high + 1):
            sumx_x = sumx(x)
            rem = D - sumx_x
            if rem >= 0 and rem >= sumy_med:
                total += count_valid_y(rem)
    
    if n > 0 and region3_x_start <= region3_x_end:
        for x in range(region3_x_start, region3_x_end + 1):
            sumx_x = sumx(x)
            rem = D - sumx_x
            if rem >= 0 and rem >= sumy_med:
                total += count_valid_y(rem)
    
    print(total)

if __name__ == "__main__":
    main()