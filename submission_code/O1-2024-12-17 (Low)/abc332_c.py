def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]

    # We'll break the schedule into segments separated by '0' days.
    # Within each segment, no washing occurs, so T-shirts cannot be reused.
    # Let a = number of '1' (meal) days in the segment
    # and b = number of '2' (event) days in the segment.
    # Takahashi has M plain shirts and we'll buy L logo shirts.
    # In a single segment:
    #  - The b = #event days MUST use logo shirts.
    #  - The a = #meal days can use either plain or logo shirts.
    #  - We can't reuse the same T-shirt in the same segment.
    #
    # For a single segment to be feasible with L logo shirts:
    #   - We must allocate exactly b event days to b distinct logo shirts.
    #   - We can allocate x meal days to logo shirts and a - x meal days to plain shirts.
    #   - We need x + b <= L   (because we can't reuse the same logo shirt)
    #   - We need (a - x) <= M (because we can't reuse the same plain shirt)
    # So x must satisfy:
    #     a - M <= x <= L - b
    # For feasibility, we require:
    #     a - M <= L - b
    # =>  L >= b + max(0, a - M)
    #
    # Hence, the minimal L needed for one segment is L_seg = b + max(0, a - M).
    # Overall, we need L >= max(L_seg) over all segments.
    # Then the answer is that maximum (or 0 if that max is negative, but it won't be).

    segments = []
    i = 0
    while i < N:
        if S[i] == '0':
            i += 1
            continue
        start = i
        while i < N and S[i] != '0':
            i += 1
        segments.append(S[start:i])

    answer = 0
    for seg in segments:
        a = seg.count('1')
        b = seg.count('2')
        needed = b + max(0, a - M)
        if needed > answer:
            answer = needed

    print(answer if answer > 0 else 0)

# Do not forget to call main()
if __name__ == "__main__":
    main()