def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input())
    A = list(map(int, input().split()))
    
    # build list of positions for each value 1..N
    pos = [[] for _ in range(N+1)]
    for i, a in enumerate(A, start=1):
        pos[a].append(i)
    
    total = 0
    # for x from 1..N, we count subarrays where x is present
    # and x-1 is absent.
    # Positions of forbidden value (x-1) split [1..N] into segments
    # without x-1; inside each segment, count subarrays containing at least one x.
    for x in range(1, N+1):
        forb = pos[x-1] if x > 1 else []
        # make boundaries
        # t0 = 0, t_{m+1} = N+1
        # segments [t_j+1 .. t_{j+1}-1] have no x-1
        last_forb = 0
        fi = 0
        f_len = len(forb)
        # iterate over each forb position, and then one more at N+1
        while fi < f_len:
            b = forb[fi]
            l = last_forb + 1
            r = b - 1
            if l <= r:
                # process segment [l..r]
                # get occurrences of x in this segment
                # pos[x] is sorted; we can advance pointer
                # but simpler: use binary search to slice
                import bisect
                p_list = pos[x]
                lo = bisect.bisect_left(p_list, l)
                hi = bisect.bisect_right(p_list, r)
                prev_x = l - 1
                for pj in p_list[lo:hi]:
                    left = pj - prev_x
                    right = r - pj + 1
                    total += left * right
                    prev_x = pj
            last_forb = b
            fi += 1
        # final segment after last forb
        l = last_forb + 1
        r = N
        if l <= r:
            import bisect
            p_list = pos[x]
            lo = bisect.bisect_left(p_list, l)
            hi = bisect.bisect_right(p_list, r)
            prev_x = l - 1
            for pj in p_list[lo:hi]:
                left = pj - prev_x
                right = r - pj + 1
                total += left * right
                prev_x = pj
    
    print(total)

if __name__ == "__main__":
    main()