import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N, D = int(input[idx]), int(input[idx+1])
    idx +=2
    x = []
    y = []
    for _ in range(N):
        xi = int(input[idx])
        yi = int(input[idx+1])
        x.append(xi)
        y.append(yi)
        idx +=2
    
    x.sort()
    y.sort()
    
    prefix_x = [0]*(N+1)
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + x[i]
    
    prefix_y = [0]*(N+1)
    for i in range(N):
        prefix_y[i+1] = prefix_y[i] + y[i]
    
    def compute_sum_x(x_val):
        k = bisect.bisect_right(x, x_val)
        return x_val * k - prefix_x[k] + (prefix_x[N] - prefix_x[k]) - x_val * (N - k)
    
    median_x = x[N//2]
    sum_x_min = compute_sum_x(median_x)
    if sum_x_min > D:
        print(0)
        return
    
    low = -10**18
    high = median_x
    while low < high:
        mid = (low + high) // 2
        if compute_sum_x(mid) <= D:
            high = mid
        else:
            low = mid + 1
    x_min = low
    
    low = median_x
    high = 10**18
    while low < high:
        mid = (low + high + 1) // 2
        if compute_sum_x(mid) <= D:
            low = mid
        else:
            high = mid -1
    x_max = low
    
    total = 0
    
    def compute_sum_y(y_val):
        k = bisect.bisect_right(y, y_val)
        return y_val * k - prefix_y[k] + (prefix_y[N] - prefix_y[k]) - y_val * (N - k)
    
    median_y = y[N//2]
    sum_y_min = compute_sum_y(median_y)
    
    for x_val in range(x_min, x_max + 1):
        s_x = compute_sum_x(x_val)
        K = D - s_x
        if K < 0:
            continue
        
        if K < sum_y_min:
            continue
        
        low = -10**18
        high = median_y
        y_a = None
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum_y(mid)
            if s <= K:
                y_a = mid
                high = mid -1
            else:
                low = mid +1
        
        if y_a is None:
            continue
        
        low = median_y
        high = 10**18
        y_b = None
        while low <= high:
            mid = (low + high + 1) // 2
            s = compute_sum_y(mid)
            if s <= K:
                y_b = mid
                low = mid +1
            else:
                high = mid -1
        
        if y_b is None:
            continue
        
        cnt = max(0, y_b - y_a + 1)
        total += cnt
    
    print(total)

if __name__ == "__main__":
    main()