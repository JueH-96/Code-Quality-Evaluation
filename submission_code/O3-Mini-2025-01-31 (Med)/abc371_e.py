def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))

    # Build a dictionary mapping each value to the list of indices (1-indexed) where it appears.
    pos = {}
    for i, a in enumerate(A, start=1):
        if a not in pos:
            pos[a] = []
        pos[a].append(i)

    total_intervals = n * (n + 1) // 2

    # The sum we need equals the sum over distinct x in A of the number of intervals that contain x.
    # For a given x with positions [p1, p2, ..., pk], the intervals that do NOT contain x are
    # those that lie entirely in one of the gaps:
    #   before the first occurrence: from index 1 to p1-1,
    #   between consecutive occurrences: from p_i+1 to p_(i+1)-1,
    #   after the last occurrence: from p_k+1 to n.
    # The number of intervals inside a gap of length L is L*(L+1)//2.
    # Thus, the number of intervals that DO contain x is:
    #     total_intervals - (sum of intervals in all gaps).
    result = 0
    for a, indices in pos.items():
        gap_sum = 0
        # Gap before the first occurrence.
        gap = indices[0] - 1
        gap_sum += gap * (gap + 1) // 2
        
        # Gaps between consecutive occurrences.
        for i in range(len(indices) - 1):
            gap = indices[i+1] - indices[i] - 1
            gap_sum += gap * (gap + 1) // 2
        
        # Gap after the last occurrence.
        gap = n - indices[-1]
        gap_sum += gap * (gap + 1) // 2
        
        intervals_with_a = total_intervals - gap_sum
        result += intervals_with_a

    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()