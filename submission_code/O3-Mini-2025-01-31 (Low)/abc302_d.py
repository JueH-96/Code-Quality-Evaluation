def main():
    import sys, bisect

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    D = int(next(it))
    
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # Sort both lists.
    A.sort()
    B.sort()
    
    # We'll iterate through A (choosing a gift for Aoki), and then
    # use binary search on B (gifts for Snuke) to find gifts such that
    # |a - b| <= D.
    # We want to maximize a + b and B is sorted. For a fixed a, 
    # the best candidate b is the largest b such that:
    #   a - D <= b <= a + D.
    
    ans = -1
    for a in reversed(A):
        low = a - D
        high = a + D
        # Use bisect to find the first index in B with value >= low:
        left_index = bisect.bisect_left(B, low)
        # Use bisect_right to get one past the index of the last element <= high:
        right_index = bisect.bisect_right(B, high) - 1
        if left_index <= right_index and 0 <= right_index < M:
            candidate = a + B[right_index]
            if candidate > ans:
                ans = candidate
    
    sys.stdout.write(str(ans))
    
# Call main()    
if __name__ == '__main__':
    main()