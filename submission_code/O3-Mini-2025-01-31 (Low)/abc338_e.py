def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # There are 2*n distinct endpoints on positions 1..2*n.
    # Each chord connects two distinct positions on the circle.
    # When chords are drawn on a circle, two chords intersect if and only
    # if their endpoints appear in alternating order.
    # One standard strategy is to "sweep" the circle in order (since the 2*n points
    # are arranged in clockwise order from 1 to 2*n) and use a Fenwick tree
    # (binary indexed tree) to detect an "inversion" pattern.
    # Specifically, label each endpoint with the chord id it belongs to.
    # Then, while scanning positions 1,2,...,2*n:
    #   - When you see a chord for the first time, store its position (and update BIT).
    #   - When you see it the second time, query the BIT for any first endpoint of another
    #     chord that was placed strictly between this chord's first endpoint and the current position.
    # If the query returns a count > 0 then that chord’s endpoints interlace with the current chord,
    # meaning the chords intersect.
    #
    # Read the chords and build an order array P where P[i] is the chord id whose endpoint is at i+1.
    P = [None] * (2 * n)
    for chord_id in range(n):
        a = int(next(it))
        b = int(next(it))
        # Place chord_id at positions a and b (adjusting for 0-indexing)
        P[a - 1] = chord_id
        P[b - 1] = chord_id

    # Fenwick tree implementation for 1-indexed positions
    size = 2 * n
    fenw = [0] * (size + 1)
    
    def fenw_update(i, delta):
        while i <= size:
            fenw[i] += delta
            i += i & -i

    def fenw_sum(i):
        s = 0
        while i:
            s += fenw[i]
            i -= i & -i
        return s

    # Dictionary to store for each chord id the position of its first occurrence.
    firstOccurrence = {}
    # Traverse the circle in order (positions 1 to 2*n).
    for pos_idx, chord in enumerate(P, start=1):
        if chord not in firstOccurrence:
            # First occurrence of this chord.
            firstOccurrence[chord] = pos_idx
            # Mark it active in the BIT
            fenw_update(pos_idx, 1)
        else:
            # Second occurrence. Let start be its first occurrence.
            start = firstOccurrence[chord]
            # Query how many active chords had their first endpoint between start and current pos_idx.
            cnt = fenw_sum(pos_idx - 1) - fenw_sum(start)
            if cnt > 0:
                sys.stdout.write("Yes")
                return
            # Remove this chord's marker from the BIT.
            fenw_update(start, -1)
    sys.stdout.write("No")


if __name__ == '__main__':
    main()