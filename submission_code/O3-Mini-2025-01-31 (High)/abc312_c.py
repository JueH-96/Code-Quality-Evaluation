def main():
    import sys
    import bisect

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    A.sort()  # Sellers: each seller i is willing if X >= A_i
    B.sort()  # Buyers: each buyer i is willing if X <= B_i
    
    # We want to find the minimum X such that:
    #   (# sellers with A_i <= X) >= (# buyers with B_i >= X)
    # Let f(X) = (# sellers with A_i <= X) - (# buyers with B_i >= X)
    # Notice that f(X) is a non-decreasing function.
    # We can use binary search for the smallest integer X for which f(X) >= 0.
    
    # Set search interval. For X greater than max(B), no buyer will be willing, so condition always holds.
    low = 0
    high = max(max(A), max(B)) + 1
    
    while low < high:
        mid = (low + high) // 2
        # Count sellers willing to sell at price mid.
        sellers_count = bisect.bisect_right(A, mid)
        # Count buyers willing to buy at price mid.
        # Buyers are willing if B_i >= mid. With sorted B, we can do:
        buyers_idx = bisect.bisect_left(B, mid)
        buyers_count = M - buyers_idx
        
        if sellers_count >= buyers_count:
            high = mid
        else:
            low = mid + 1

    print(low)

if __name__ == '__main__':
    main()