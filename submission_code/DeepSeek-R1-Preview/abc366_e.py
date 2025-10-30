import bisect

def compute_sx(x, x_list, prefix):
    n = len(x_list)
    idx = bisect.bisect_right(x_list, x)
    sum_left = prefix[idx]
    sum_right = prefix[-1] - sum_left
    s = x * idx - sum_left + (sum_right - x * (n - idx))
    return s

def compute_sy(y, y_list, prefix):
    n = len(y_list)
    idx = bisect.bisect_right(y_list, y)
    sum_left = prefix[idx]
    sum_right = prefix[-1] - sum_left
    s = y * idx - sum_left + (sum_right - y * (n - idx))
    return s

def find_x_left(x_list, prefix, D):
    low = -10**18
    high = 10**18
    while low < high:
        mid = (low + high) // 2
        s = compute_sx(mid, x_list, prefix)
        if s <= D:
            high = mid
        else:
            low = mid + 1
    if compute_sx(low, x_list, prefix) <= D:
        return low
    else:
        return None

def find_x_right(x_list, prefix, D):
    low = -10**18
    high = 10**18
    while low < high:
        mid = (low + high + 1) // 2
        s = compute_sx(mid, x_list, prefix)
        if s <= D:
            low = mid
        else:
            high = mid - 1
    if compute_sx(low, x_list, prefix) <= D:
        return low
    else:
        return None

def find_y_left(y_list, prefix, D):
    low = -10**18
    high = 10**18
    while low < high:
        mid = (low + high) // 2
        s = compute_sy(mid, y_list, prefix)
        if s <= D:
            high = mid
        else:
            low = mid + 1
    if compute_sy(low, y_list, prefix) <= D:
        return low
    else:
        return None

def find_y_right(y_list, prefix, D):
    low = -10**18
    high = 10**18
    while low < high:
        mid = (low + high + 1) // 2
        s = compute_sy(mid, y_list, prefix)
        if s <= D:
            low = mid
        else:
            high = mid - 1
    if compute_sy(low, y_list, prefix) <= D:
        return low
    else:
        return None

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
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        x_list.append(x)
        y_list.append(y)
    
    # Compute for x
    x_list.sort()
    prefix_x = [0] * (len(x_list) + 1)
    for i in range(len(x_list)):
        prefix_x[i + 1] = prefix_x[i] + x_list[i]
    x_left = find_x_left(x_list, prefix_x, D)
    if x_left is None:
        print(0)
        return
    x_right = find_x_right(x_list, prefix_x, D)
    if x_right is None or x_right < x_left:
        print(0)
        return
    
    # Generate s_x_list
    s_x_list = []
    for x in range(x_left, x_right + 1):
        s = compute_sx(x, x_list, prefix_x)
        s_x_list.append(s)
    
    # Compute for y
    y_list.sort()
    prefix_y = [0] * (len(y_list) + 1)
    for i in range(len(y_list)):
        prefix_y[i + 1] = prefix_y[i] + y_list[i]
    y_left = find_y_left(y_list, prefix_y, D)
    if y_left is None:
        print(0)
        return
    y_right = find_y_right(y_list, prefix_y, D)
    if y_right is None or y_right < y_left:
        print(0)
        return
    
    # Generate s_y_list
    s_y_list = []
    for y in range(y_left, y_right + 1):
        s = compute_sy(y, y_list, prefix_y)
        s_y_list.append(s)
    
    # Sort the lists
    s_x_list.sort()
    s_y_list.sort()
    
    # Two-pointer approach
    count = 0
    j = len(s_y_list) - 1
    for s_x in s_x_list:
        while j >= 0 and s_x + s_y_list[j] > D:
            j -= 1
        count += (j + 1)
    
    print(count)

if __name__ == '__main__':
    main()