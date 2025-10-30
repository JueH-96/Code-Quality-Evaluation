import bisect

def compute_sum(sorted_arr, prefix, total, val):
    n = len(sorted_arr)
    idx = bisect.bisect_right(sorted_arr, val) - 1
    count = idx + 1
    sum_left = val * count - prefix[idx + 1]
    sum_right = (total - prefix[idx + 1]) - val * (n - count)
    return sum_left + sum_right

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    D = int(input[idx]); idx += 1
    xs = []
    ys = []
    for _ in range(N):
        x = int(input[idx]); idx += 1
        y = int(input[idx]); idx += 1
        xs.append(x)
        ys.append(y)

    # Process x component
    sorted_x = sorted(xs)
    n = len(sorted_x)
    prefix_x = [0] * (n + 1)
    for i in range(n):
        prefix_x[i+1] = prefix_x[i] + sorted_x[i]
    total_x = prefix_x[-1]

    # Compute median_x
    if n % 2 == 1:
        median_x = sorted_x[n//2]
    else:
        median_x = (sorted_x[n//2 - 1] + sorted_x[n//2]) // 2

    # Compute minimal sum_x
    Sx_min = compute_sum(sorted_x, prefix_x, total_x, median_x)
    if D < Sx_min:
        print(0)
        return

    # Find x_low and x_high
    current_x = median_x
    current_sum = Sx_min
    x_low = current_x
    while True:
        next_x = current_x - 1
        next_sum = compute_sum(sorted_x, prefix_x, total_x, next_x)
        if next_sum > D:
            break
        current_x = next_x
        x_low = current_x

    current_x = median_x
    x_high = current_x
    while True:
        next_x = current_x + 1
        next_sum = compute_sum(sorted_x, prefix_x, total_x, next_x)
        if next_sum > D:
            break
        current_x = next_x
        x_high = current_x

    # Process y component
    sorted_y = sorted(ys)
    m = len(sorted_y)
    prefix_y = [0] * (m + 1)
    for i in range(m):
        prefix_y[i+1] = prefix_y[i] + sorted_y[i]
    total_y = prefix_y[-1]

    if m % 2 == 1:
        median_y = sorted_y[m//2]
    else:
        median_y = (sorted_y[m//2 - 1] + sorted_y[m//2]) // 2

    total = 0

    # Iterate over all x in [x_low, x_high]
    for x in range(x_low, x_high + 1):
        Sx_val = compute_sum(sorted_x, prefix_x, total_x, x)
        L = D - Sx_val
        if L < 0:
            continue

        # Compute number of y's with Sy(y) <= L
        current_sum_y = compute_sum(sorted_y, prefix_y, total_y, median_y)
        if current_sum_y > L:
            continue

        # Find left boundary
        current_y = median_y
        left_boundary = current_y
        while True:
            next_y = current_y - 1
            next_sum = compute_sum(sorted_y, prefix_y, total_y, next_y)
            if next_sum > L:
                break
            current_y = next_y
            left_boundary = current_y

        # Find right boundary
        current_y = median_y
        right_boundary = current_y
        while True:
            next_y = current_y + 1
            next_sum = compute_sum(sorted_y, prefix_y, total_y, next_y)
            if next_sum > L:
                break
            current_y = next_y
            right_boundary = current_y

        count = right_boundary - left_boundary + 1
        total += count

    print(total)

if __name__ == "__main__":
    main()