def main():
    import sys, bisect
    input = sys.stdin.readline
    
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))
    A.sort()
    
    # For each query: b, k: we want to find the minimal d such that there are at least k points with |a - b| <= d.
    # For this, define function count_points(b, d) = number of points in A s.t. |a - b| <= d.
    def count_points(b, d):
        # we need count of a in [b-d, b+d]
        left = bisect.bisect_left(A, b - d)
        right = bisect.bisect_right(A, b + d)
        return right - left

    results = []
    for _ in range(Q):
        b, k = map(int, input().split())
        lo = 0
        hi = 200000000  # since |a - b| can be at most 2e8
        while lo < hi:
            mid = (lo + hi) // 2
            if count_points(b, mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        results.append(lo)
    
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()