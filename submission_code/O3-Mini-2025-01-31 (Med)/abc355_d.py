def main():
    import sys
    import bisect

    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    intervals = []
    index = 1
    for _ in range(n):
        l = int(input_data[index])
        r = int(input_data[index+1])
        index += 2
        intervals.append((l, r))

    # Create a sorted list of all left endpoints.
    lefts = sorted(l for l, _ in intervals)

    # Count disjoint pairs.
    # Two intervals i and j (with i having the smaller ending point)
    # will be disjoint if r_i < l_j.
    disjoint_pairs = 0
    for l, r in intervals:
        # Find the first index in lefts where the left endpoint is greater than r.
        pos = bisect.bisect_right(lefts, r)
        # Every interval starting from this position has l > r.
        disjoint_pairs += (n - pos)
    
    # Total pairs.
    total_pairs = n * (n - 1) // 2
    # The answer: number of intersecting pairs.
    print(total_pairs - disjoint_pairs)

if __name__ == '__main__':
    main()