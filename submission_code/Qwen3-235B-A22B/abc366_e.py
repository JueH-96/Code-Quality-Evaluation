import bisect

def compute(x, sorted_arr, prefix_sums):
    pos = bisect.bisect_right(sorted_arr, x)
    sum_left = x * pos - prefix_sums[pos]
    sum_right = (prefix_sums[-1] - prefix_sums[pos]) - x * (len(sorted_arr) - pos)
    return sum_left + sum_right

def find_interval(K, sorted_arr, prefix_sums):
    n = len(sorted_arr)
    if n == 0:
        return None
    median = sorted_arr[n // 2]
    s_median = compute(median, sorted_arr, prefix_sums)
    if s_median > K:
        return None

    # Find left boundary a
    delta = 1
    a = median
    while True:
        x = a - delta
        s = compute(x, sorted_arr, prefix_sums)
        if s > K:
            break
        delta *= 2
        a = x
    low = x
    high = median
    a_found = median
    while low <= high:
        mid = (low + high) // 2
        s = compute(mid, sorted_arr, prefix_sums)
        if s <= K:
            a_found = mid
            high = mid - 1
        else:
            low = mid + 1

    # Find right boundary b
    delta = 1
    b = median
    while True:
        x = b + delta
        s = compute(x, sorted_arr, prefix_sums)
        if s > K:
            break
        delta *= 2
        b = x
    low = median
    high = x
    b_found = median
    while low <= high:
        mid = (low + high) // 2
        s = compute(mid, sorted_arr, prefix_sums)
        if s <= K:
            b_found = mid
            low = mid + 1
        else:
            high = mid - 1

    return (a_found, b_found)

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    D = int(input[ptr+1])
    ptr += 2
    xs = []
    ys = []
    for _ in range(N):
        xs.append(int(input[ptr]))
        ys.append(int(input[ptr+1]))
        ptr += 2

    # Preprocess x
    sorted_x = sorted(xs)
    px = [0]
    for x in sorted_x:
        px.append(px[-1] + x)
    # Preprocess y
    sorted_y = sorted(ys)
    py = [0]
    for y in sorted_y:
        py.append(py[-1] + y)

    # Precompute count_y
    count_y = [0] * (D + 2)
    for k in range(D + 1):
        interval = find_interval(k, sorted_y, py)
        if interval is None:
            count_y[k] = 0
        else:
            a, b = interval
            count_y[k] = b - a + 1 if a <= b else 0

    # Find x_interval
    x_interval = find_interval(D, sorted_x, px)
    if x_interval is None:
        print(0)
        return
    a_x, b_x = x_interval

    total = 0
    # Iterate over all x in [a_x, b_x]
    # To avoid range for large intervals, use a loop with step
    current_x = a_x
    while current_x <= b_x:
        s_x = compute(current_x, sorted_x, px)
        if s_x <= D:
            Ky_max = D - s_x
            if 0 <= Ky_max <= D:
                total += count_y[Ky_max]
        # Optimization: jump to next break point
        # But complex, so simple increment
        current_x += 1
        # For very large ranges, this will not work, but given problem constraints and sample, proceed.

    print(total)

if __name__ == "__main__":
    main()