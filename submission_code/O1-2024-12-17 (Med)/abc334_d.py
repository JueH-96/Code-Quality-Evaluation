def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    R = list(map(int, input_data[2:2+N]))
    queries = list(map(int, input_data[2+N:]))

    R.sort()
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + R[i - 1]

    # For each query X, find the maximum k such that prefix_sums[k] <= X
    idx = 0
    for _ in range(Q):
        X = queries[idx]
        idx += 1
        # bisect_right returns the insertion position to the right of X in sorted order
        # We want the largest i such that prefix_sums[i] <= X => i = bisect_right(prefix_sums, X) - 1
        answer = bisect.bisect_right(prefix_sums, X) - 1
        # Ensure answer is within 0..N
        if answer < 0:
            answer = 0
        elif answer > N:
            answer = N
        print(answer)

# Do not forget to call main
if __name__ == "__main__":
    main()