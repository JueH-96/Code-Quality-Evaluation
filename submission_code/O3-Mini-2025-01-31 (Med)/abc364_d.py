def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    A.sort()  # sort the points A_1, ..., A_N

    out_lines = []
    
    # For each query, perform a binary search on d: the candidate distance.
    # We want the smallest d such that at least k points satisfy |A_i - b| <= d.
    for _ in range(q):
        b = int(next(it))
        k = int(next(it))
        
        lo, hi = 0, 200000000  # maximum possible distance
        while lo <= hi:
            mid = (lo + hi) // 2
            # Count how many points A_i are within distance mid from b, i.e.
            # such that A_i is in the interval [b - mid, b + mid]
            l_bound = b - mid
            r_bound = b + mid
            count = bisect.bisect_right(A, r_bound) - bisect.bisect_left(A, l_bound)
            
            if count >= k:
                hi = mid - 1
            else:
                lo = mid + 1
        # After binary search, lo is the smallest d meeting the condition.
        out_lines.append(str(lo))
    
    sys.stdout.write("
".join(out_lines))


if __name__ == '__main__':
    main()