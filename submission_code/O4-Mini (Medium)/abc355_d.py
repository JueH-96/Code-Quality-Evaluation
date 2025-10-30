import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        return
    N = int(n_line)
    intervals = []
    Rs = [0]*N
    for i in range(N):
        l, r = map(int, data.readline().split())
        intervals.append((l, r))
        Rs[i] = r

    # Sort intervals by left endpoint
    intervals.sort(key=lambda x: x[0])

    # Coordinate-compress the r-values
    Rs_unique = sorted(set(Rs))
    m = len(Rs_unique)
    # Map r -> its 1-based index in BIT
    comp = {v: i+1 for i, v in enumerate(Rs_unique)}

    # Fenwick / BIT for counts of r's seen so far
    BIT = [0] * (m+1)
    def bit_add(i, v):
        # i in [1..m]
        while i <= m:
            BIT[i] += v
            i += i & -i
    def bit_sum(i):
        # sum of [1..i]
        s = 0
        while i > 0:
            s += BIT[i]
            i -= i & -i
        return s

    import bisect
    non_intersect = 0
    # Process each interval in increasing l
    for l, r in intervals:
        # count how many previous r_i < l
        # find # of compressed r-values < l
        # bisect_left in Rs_unique
        pos = bisect.bisect_left(Rs_unique, l)
        if pos > 0:
            non_intersect += bit_sum(pos)
        # add this interval's r to BIT
        bit_add(comp[r], 1)

    # total pairs
    total_pairs = N*(N-1)//2
    # intersecting = total_pairs - non-intersecting
    print(total_pairs - non_intersect)

if __name__ == "__main__":
    main()