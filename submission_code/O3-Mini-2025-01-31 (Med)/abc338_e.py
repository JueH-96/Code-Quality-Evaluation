def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Each chord is given by two endpoints on a circle.
    # We “normalize” each chord so that the first endpoint is the smaller number.
    # (This is safe if we cut the circle at a point not used by any chord – for example, between 2N and 1.)
    chords = []
    for i in range(n):
        a = int(next(it))
        b = int(next(it))
        if a < b:
            chords.append((a, b))
        else:
            chords.append((b, a))
    total = 2 * n
    # Create an array of events for positions 1..total.
    # Each endpoint occurs exactly once.
    # For each chord i with endpoints (a, b) (with a < b):
    #   – at position a, we mark a "start" event for chord i
    #   – at position b, we mark an "end" event for chord i.
    events = [None] * (total + 1)  # positions 1-indexed
    for i, (a, b) in enumerate(chords):
        events[a] = (i, True)   # start event
        events[b] = (i, False)  # end event

    # We will use a Fenwick tree (Binary Indexed Tree)
    # to help answer queries of the form: “how many chords have started (and are active)
    # between two positions?”
    size = total
    fenw = [0] * (size + 1)
    def update(i, delta):
        while i <= size:
            fenw[i] += delta
            i += i & -i

    def prefix_sum(i):
        s = 0
        while i:
            s += fenw[i]
            i -= i & -i
        return s

    def range_sum(l, r):
        if r < l:
            return 0
        return prefix_sum(r) - prefix_sum(l - 1)

    # Explanation of the detection method:
    # It is a known fact in circle chord problems that two chords (a, b) and (c, d)
    # (with a < c) intersect if and only if a < c < b < d.
    # We simulate a sweep along the circle from 1 to 2N.
    # When we see the first (start) endpoint of a chord, we “activate” that chord by inserting its start position into the BIT.
    # When we later see the ending endpoint of a chord (say chord i with start position a and end position b),
    # we then query the BIT to see if there is any chord that started strictly between a and b and which has not yet ended.
    # Note that in a non-intersecting configuration (think of a “well‐formed parentheses” or nested intervals scenario),
    # any chord that starts after a must finish (end) before chord i ends.
    # Thus if we discover any active chord that started after a when chord i is closing, then its second endpoint is still in the future,
    # and hence we have a pair of chords with interwoven endpoints: a < (start of other) < b < (end of other).
    # In other words, they intersect.
    first_occurrence = [0] * n

    for pos in range(1, total + 1):
        chord_id, is_start = events[pos]
        if is_start:
            # When a chord starts, record its start position and mark it as active.
            first_occurrence[chord_id] = pos
            update(pos, 1)
        else:
            # At the end event of chord chord_id (with start position start_pos and end = pos),
            # query for any active chord that started strictly between start_pos and pos.
            start_pos = first_occurrence[chord_id]
            cnt = range_sum(start_pos + 1, pos - 1)
            if cnt > 0:
                sys.stdout.write("Yes")
                return
            # Remove the chord from the BIT.
            update(start_pos, -1)
    sys.stdout.write("No")


if __name__ == '__main__':
    main()