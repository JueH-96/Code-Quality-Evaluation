def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    A.sort()

    # Precompute minimum and maximum of A to limit search range
    minA = A[0]
    maxA = A[-1]

    # Helper function to count how many points in A are within distance d of b
    def count_within_distance(b, d):
        left = bisect.bisect_left(A, b - d)
        right = bisect.bisect_right(A, b + d)
        return right - left

    idx = 2 + N
    results = []
    for _ in range(Q):
        b = int(input_data[idx]); idx += 1
        k = int(input_data[idx]); idx += 1

        # The maximum possible distance to any point in A for the current b
        high = max(abs(b - minA), abs(b - maxA))
        low = 0

        # Binary search for the k-th smallest distance
        while low < high:
            mid = (low + high) // 2
            if count_within_distance(b, mid) >= k:
                high = mid
            else:
                low = mid + 1

        results.append(str(low))

    print("
".join(results))

# Do not forget to call main at the end
if __name__ == "__main__":
    main()