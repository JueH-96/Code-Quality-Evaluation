def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse the first line
    N, Q = map(int, input_data[:2])
    
    # Parse the coordinates of A
    a = list(map(int, input_data[2:2+N]))
    # Sort A for efficient distance counting
    a.sort()
    
    # We'll read the queries (b_j, k_j)
    queries = input_data[2+N:]
    
    # Precompute minA and maxA for bounding the binary search range
    minA, maxA = a[0], a[-1]

    # Function to count how many points a_i satisfy |a_i - b| <= d
    def count_within_distance(b, d):
        left = bisect.bisect_left(a, b - d)
        right = bisect.bisect_right(a, b + d)
        return right - left
    
    # Process each query
    idx = 0
    out = []
    for _ in range(Q):
        b_j = int(queries[idx]); idx += 1
        k_j = int(queries[idx]); idx += 1

        # Distance cannot exceed the farthest endpoint from b_j
        dist_max = max(abs(b_j - minA), abs(b_j - maxA))

        # Binary search for the k_j-th smallest distance
        lo, hi = 0, dist_max
        while lo < hi:
            mid = (lo + hi) // 2
            if count_within_distance(b_j, mid) >= k_j:
                hi = mid
            else:
                lo = mid + 1
        
        out.append(str(lo))
    
    # Print the answers
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()