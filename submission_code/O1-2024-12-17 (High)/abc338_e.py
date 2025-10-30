def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = 2 * N  # total number of distinct points on the circle

    # P[k] will store the chord index that occupies (i.e., has an endpoint at) point k
    P = [0] * (M + 1)

    # Read chord endpoints
    idx = 1
    for chord_index in range(1, N + 1):
        a = int(data[idx])
        b = int(data[idx + 1])
        idx += 2
        P[a] = chord_index
        P[b] = chord_index

    # Fenwick (Binary Indexed) Tree for "open times"
    fenwicks = [0] * (M + 1)

    def fenwicks_add(x, val):
        """Add 'val' to Fenwick tree at index x."""
        while x <= M:
            fenwicks[x] += val
            x += x & -x

    def fenwicks_sum(x):
        """Returns the prefix sum from 1 to x inclusive."""
        s = 0
        while x > 0:
            s += fenwicks[x]
            x -= x & -x
        return s

    def fenwicks_range_sum(l, r):
        """Returns the sum in Fenwick tree from index l to r inclusive."""
        return fenwicks_sum(r) - fenwicks_sum(l - 1)

    # To store the position (k) at which each chord first appears ("opens")
    open_time = [0] * (N + 1)

    # Sweep around the circle in ascending order of points 1..2N
    for k in range(1, M + 1):
        c = P[k]  # chord index for point k
        if open_time[c] == 0:
            # First occurrence => open this chord at position k
            open_time[c] = k
            fenwicks_add(k, 1)
        else:
            # Second occurrence => close this chord
            ot = open_time[c]
            # Check if there is any chord that opened after 'ot' (i.e. in range ot+1..M)
            # and is still open (not yet closed).
            if fenwicks_range_sum(ot + 1, M) > 0:
                print("Yes")
                return
            # Remove this chord's opening from Fenwicks, since it is now closed
            fenwicks_add(ot, -1)

    # If we finish without finding an intersection:
    print("No")

# Do not forget to call main() or you will receive no points.
if __name__ == "__main__":
    main()