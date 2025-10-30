def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    idx = 0

    N = int(input_data[idx]); idx += 1

    X = list(map(int, input_data[idx:idx+N]))
    idx += N

    P = list(map(int, input_data[idx:idx+N]))
    idx += N

    # Build prefix sums of populations
    prefix_sum = [0] * (N+1)
    for i in range(1, N+1):
        prefix_sum[i] = prefix_sum[i-1] + P[i-1]

    Q = int(input_data[idx]); idx += 1

    outputs = []
    for _ in range(Q):
        L, R = map(int, input_data[idx:idx+2])
        idx += 2
        # Find the leftmost and rightmost village indices in [L, R]
        # leftIndex is the first index where X[leftIndex] >= L
        leftIndex = bisect.bisect_left(X, L)
        # rightIndex is one past the last index where X[rightIndex] <= R, so we do bisect_right
        rightIndex = bisect.bisect_right(X, R)
        
        if leftIndex < rightIndex:
            # Sum from prefix_sum[leftIndex] to prefix_sum[rightIndex - 1]
            total_pop = prefix_sum[rightIndex] - prefix_sum[leftIndex]
            outputs.append(str(total_pop))
        else:
            outputs.append("0")

    print("
".join(outputs))

# Do not forget to call main()!
if __name__ == "__main__":
    main()