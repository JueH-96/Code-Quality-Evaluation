def main():
    import sys
    from bisect import bisect_left, bisect_right
    
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    
    A = [int(next(it)) for _ in range(N)]
    A.sort()
    
    # Prepare output list to join later.
    answers = []
    
    # Process each query.
    for _ in range(Q):
        b = int(next(it))
        k = int(next(it))
        
        # We need to find the smallest d such that at least k points in A have 
        # |A_i - b| <= d. In other words, count of points in [b-d, b+d] must be >= k.
        # We'll binary search for d. The maximum possible distance is when one of the points
        # is at one extreme. Since -10^8 <= a_i, b <= 10^8, maximum possible |A_i - b| is 2*10^8.
        lo = 0
        hi = 200_000_000
        
        while lo < hi:
            mid = (lo + hi) // 2
            # Count points A[i] such that b - mid <= A[i] <= b + mid.
            left_index = bisect_left(A, b - mid)
            right_index = bisect_right(A, b + mid)
            count = right_index - left_index
            if count >= k:
                hi = mid
            else:
                lo = mid + 1
        
        answers.append(str(lo))
        
    sys.stdout.write("
".join(answers))
    
if __name__ == '__main__':
    main()