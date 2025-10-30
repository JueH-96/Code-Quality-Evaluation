def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = [0]*N
    H = [0]*N

    # Read the building coordinates and heights
    idx = 1
    for i in range(N):
        X[i] = int(data[idx])
        H[i] = int(data[idx+1])
        idx += 2

    # If there is only one building, then height 0 can already see it
    if N == 1:
        print(-1)
        return

    from math import inf
    max_critical = -inf

    # We only need to ensure that (H_i - y)/X_i < (H_{i+1} - y)/X_{i+1}
    # for all consecutive i to see all buildings in ascending x order.
    # Rearranging that inequality yields:
    #   y > (H_i*X_{i+1} - H_{i+1}*X_i) / (X_{i+1} - X_i).
    # We'll compute this "critical" value for each pair and take the max.
    for i in range(N-1):
        numerator = H[i]   * X[i+1] - H[i+1] * X[i]
        denominator = X[i+1] - X[i]  # positive
        critical = numerator / denominator
        if critical > max_critical:
            max_critical = critical

    # If the maximum critical value is negative, then even from y=0
    # we are above that threshold and can see all buildings, so answer -1.
    if max_critical < 0:
        print(-1)
    else:
        # Otherwise, that maximum is the highest vantage at which
        # not all buildings are visible. Above it, all are visible.
        # Print with sufficient decimal precision.
        print(f"{max_critical:.18f}")

# Do not forget to call main!
if __name__ == "__main__":
    main()