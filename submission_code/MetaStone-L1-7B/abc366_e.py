import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    D = int(data[idx])
    idx += 1
    
    x_list = []
    y_list = []
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx+1])
        x_list.append(x)
        y_list.append(y)
        idx += 2
    
    x_list.sort()
    y_list.sort()
    
    # Compute prefix sums for x
    x_prefix = [0]
    for x in x_list:
        x_prefix.append(x_prefix[-1] + x)
    
    # Compute prefix sums for y
    y_prefix = [0]
    for y in y_list:
        y_prefix.append(y_prefix[-1] + y)
    
    # Function to compute Sy for a given y
    def compute_sy(y):
        n = len(y_list)
        i = bisect.bisect_left(y_list, y)
        sum_left = y * i - y_prefix[i]
        sum_right = y_prefix[n] - y_prefix[i] - y * (n - i)
        return sum_left + sum_right
    
    # Function to compute Sx for a given x
    def compute_sx(x):
        n = len(x_list)
        i = bisect.bisect_left(x_list, x)
        sum_left = x * i - x_prefix[i]
        sum_right = x_prefix[n] - x_prefix[i] - x * (n - i)
        return sum_left + sum_right
    
    Sy_min = compute_sy(y_list[0])
    
    count = 0
    
    x_med_low = x_list[len(x_list)//2 - 1] if len(x_list) % 2 else x_list[len(x_list)//2]
    x_med_high = x_list[len(x_list)//2] if len(x_list) % 2 else x_list[len(x_list)//2 - 1]
    
    y_med_low = y_list[len(y_list)//2 - 1] if len(y_list) % 2 else y_list[len(y_list)//2]
    y_med_high = y_list[len(y_list)//2] if len(y_list) % 2 else y_list[len(y_list)//2 - 1]
    
    for x in x_list:
        Sx = compute_sx(x)
        if Sx > D:
            continue
        K = D - Sx
        if K < Sy_min:
            continue
        t = (K - Sy_min) // N
        y_low = y_med_low - t
        y_high = y_med_high + t
        
        # Find the first y >= y_low where Sy(y) <= K
        i = bisect.bisect_left(y_list, y_low)
        sy_i = compute_sy(y_list[i])
        if sy_i > K:
            continue
        
        # Find the last y <= y_high where Sy(y) <= K
        j = bisect.bisect_right(y_list, y_high) - 1
        if j < 0:
            continue
        sy_j = compute_sy(y_list[j])
        if sy_j > K:
            continue
        
        count += j - i + 1
    
    print(count)

if __name__ == '__main__':
    main()