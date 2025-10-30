def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    T = int(input_data[1])
    S = input_data[2]
    X = list(map(int, input_data[3:]))

    # Ants: (initial_position, direction, index_in_sorted_order)
    # direction: +1 if '1' (right), -1 if '0' (left)
    ants = []
    for i in range(N):
        direction = 1 if S[i] == '1' else -1
        ants.append((X[i], direction))

    # 1) Sort by initial position
    ants.sort(key=lambda x: x[0])

    # 2) Compute final positions (as floats). Add a tiny offset for ants going right
    #    so that if two ants exactly meet at t = T+0.1, we count that crossing.
    final_positions = []
    add_time = T + 0.1
    EPS = 1e-9
    for i in range(N):
        xi, vi = ants[i]
        fp = xi + vi * add_time
        if vi == 1:
            fp += EPS  # break ties in favor of "going right" being slightly larger
        final_positions.append(fp)

    # 3) Coordinate-compress these final positions
    sorted_fp = sorted(set(final_positions))
    comp_index = {}
    for i, val in enumerate(sorted_fp):
        comp_index[val] = i + 1  # Fenwick will be 1-based

    # Fenwick (BIT) for counting how many ants-going-right we have at or before a certain index
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        def update(self, i, v):
            while i <= self.n:
                self.data[i] += v
                i += i & -i
        def query(self, i):
            s = 0
            while i > 0:
                s += self.data[i]
                i -= i & -i
            return s
        def range_query(self, l, r):
            return self.query(r) - self.query(l-1)

    fenw = Fenwick(len(sorted_fp))

    # 4) Sweep ants from left to right (by initial X).
    #    - If an ant goes right (+1), we "add" its final position to Fenwick.
    #    - If an ant goes left (-1), we query how many right-ants have final_pos >= its final_pos.
    #      i.e. fenw.range_query(idx, end).
    result = 0
    for i in range(N):
        xi, vi = ants[i]
        fp = final_positions[i]
        cidx = comp_index[fp]
        if vi == 1:
            # Mark this final position in Fenwick
            fenw.update(cidx, 1)
        else:
            # Count how many "right" ants so far have final_pos >= this one
            # i.e. from cidx..end
            result += fenw.range_query(cidx, len(sorted_fp))

    print(result)

# Do not forget to call main!
if __name__ == "__main__":
    main()