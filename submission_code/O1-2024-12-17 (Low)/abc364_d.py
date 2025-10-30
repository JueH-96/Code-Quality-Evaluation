def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N, Q = map(int, input_data[:2])
    a = list(map(int, input_data[2:2+N]))
    # Each query has b_j and k_j, total 2*Q values
    queries = input_data[2+N:]
    
    # Sort the A array for efficient binary-search-based counting
    a.sort()
    
    # Predefine a function to count how many points of 'a'
    # lie in the inclusive range [low, high].
    def count_in_range(low, high):
        # Using bisect_left and bisect_right to find indices
        left_idx = bisect.bisect_left(a, low)
        right_idx = bisect.bisect_right(a, high)
        return right_idx - left_idx
    
    # We'll process each query (b, k) with a binary search on the answer (distance)
    # The maximum possible distance is up to 2*10^8 (worst case).
    # We search for the minimal T such that we have at least k points A_i with |A_i - b| <= T.
    
    idx = 0
    out = []
    for _ in range(Q):
        b_j = int(queries[idx]); idx += 1
        k_j = int(queries[idx]); idx += 1
        
        left, right = 0, 200000000  # 2*10^8
        while left < right:
            mid = (left + right) // 2
            # Count how many A_i are in [b_j - mid, b_j + mid]
            c = count_in_range(b_j - mid, b_j + mid)
            # If we have at least k_j points within distance mid, we try smaller mid
            if c >= k_j:
                right = mid
            else:
                left = mid + 1
        
        out.append(str(left))
    
    print("
".join(out))

# Do not forget to call main
if __name__ == "__main__":
    main()