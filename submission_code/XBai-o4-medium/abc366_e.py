import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    D = int(input[ptr])
    ptr += 1

    X = []
    Y = []
    for _ in range(N):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        X.append(x)
        Y.append(y)

    X.sort()
    Y.sort()

    # Compute prefix sums for X and Y
    pre_sum_x = [0] * (N + 1)
    for i in range(N):
        pre_sum_x[i+1] = pre_sum_x[i] + X[i]

    pre_sum_y = [0] * (N + 1)
    for i in range(N):
        pre_sum_y[i+1] = pre_sum_y[i] + Y[i]

    def compute_sum_x(x):
        m = bisect.bisect_right(X, x)
        return x * m - pre_sum_x[m] + (pre_sum_x[N] - pre_sum_x[m] - x * (N - m))

    def compute_sum_y(y):
        m = bisect.bisect_right(Y, y)
        return y * m - pre_sum_y[m] + (pre_sum_y[N] - pre_sum_y[m] - y * (N - m))

    def get_valid_range(sorted_vals, pre_sum):
        N_val = len(sorted_vals)
        m = N_val // 2
        current_x = sorted_vals[m]
        # Compute current_sum
        def compute_current_sum(x_val):
            m_b = bisect.bisect_right(sorted_vals, x_val)
            return x_val * m_b - pre_sum[m_b] + (pre_sum[-1] - pre_sum[m_b] - x_val * (N_val - m_b))
        current_sum = compute_current_sum(current_x)
        if current_sum > D:
            return None, None

        # Expand left
        L = current_x
        while True:
            next_x = L - 1
            m_b = bisect.bisect_right(sorted_vals, next_x)
            next_sum = next_x * m_b - pre_sum[m_b] + (pre_sum[-1] - pre_sum[m_b] - next_x * (N_val - m_b))
            if next_sum <= D:
                L = next_x
            else:
                break

        # Expand right
        R = current_x
        while True:
            next_x = R + 1
            m_b = bisect.bisect_right(sorted_vals, next_x)
            next_sum = next_x * m_b - pre_sum[m_b] + (pre_sum[-1] - pre_sum[m_b] - next_x * (N_val - m_b))
            if next_sum <= D:
                R = next_x
            else:
                break

        return L, R

    Lx, Rx = get_valid_range(X, pre_sum_x)
    if Lx is None:
        print(0)
        return

    Ly, Ry = get_valid_range(Y, pre_sum_y)
    if Ly is None:
        print(0)
        return

    from collections import defaultdict
    freq_x = defaultdict(int)
    for x in range(Lx, Rx + 1):
        s = compute_sum_x(x)
        freq_x[s] += 1

    freq_y = defaultdict(int)
    for y in range(Ly, Ry + 1):
        s = compute_sum_y(y)
        freq_y[s] += 1

    sorted_sy = sorted(freq_y.keys())
    prefix = [0]
    for sy in sorted_sy:
        prefix.append(prefix[-1] + freq_y[sy])

    ans = 0
    for sx, cntx in freq_x.items():
        if sx > D:
            continue
        remaining = D - sx
        idx = bisect.bisect_right(sorted_sy, remaining)
        ans += cntx * prefix[idx]

    print(ans)

if __name__ == "__main__":
    main()