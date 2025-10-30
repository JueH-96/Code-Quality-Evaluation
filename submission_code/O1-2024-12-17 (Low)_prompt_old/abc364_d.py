def solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    # Next N integers are a_i
    A = list(map(int, input_data[2:2+N]))
    # The remaining are b_j, k_j pairs
    queries = input_data[2+N:]
    
    A.sort()
    
    # Precompute minA, maxA for bounding the distance search
    # However, the max distance could be up to 2*10^8 (worst-case).
    # We'll just use 200,000,000 (2e8) as an upper bound for binary search.
    # We do a binary search for each query.
    
    idx = 0
    out = []
    for _ in range(Q):
        b = int(queries[idx]); idx+=1
        k = int(queries[idx]); idx+=1
        
        low, high = 0, 200000000
        
        while low < high:
            mid = (low + high) // 2
            left = b - mid
            right = b + mid
            # Count how many A's are in [left, right]
            # Using bisect on the sorted array A
            c_left = bisect.bisect_left(A, left)
            c_right = bisect.bisect_right(A, right)
            count_in_range = c_right - c_left
            
            if count_in_range >= k:
                high = mid
            else:
                low = mid + 1
        
        out.append(str(low))
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()