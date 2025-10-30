def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        N = int(data[index])
        index += 1
        # Read the sequence A (of length 2N)
        arr = list(map(int, data[index:index + 2 * N]))
        index += 2 * N

        # For each couple 1..N, record its two occurrence positions.
        # We use a list "couples" such that couples[a] = [first, second].
        couples = [[0, 0] for _ in range(N + 1)]
        # Also build an array to mark positions (1-indexed) that are first occurrences.
        # For position i (1-indexed), first_occ[i] will be the couple id if that seat is the first occurrence.
        first_occ = [None] * (2 * N + 1)
        for pos in range(1, 2 * N + 1):
            a = arr[pos - 1]
            if couples[a][0] == 0:
                couples[a][0] = pos
                first_occ[pos] = a
            else:
                couples[a][1] = pos

        # Mark couples as "valid" only if their two appearances are not adjacent.
        valid = [False] * (N + 1)
        for a in range(1, N + 1):
            # valid only if the second occurrence is not exactly one more than the first occurrence.
            if couples[a][1] - couples[a][0] > 1:
                valid[a] = True

        # Now count valid pairs (a, b) (with a < b) such that the first occurrences occur in consecutive seats
        # and the second occurrences differ by exactly 1.
        ans = 0
        # We check positions 1 through 2N-1
        for pos in range(1, 2 * N):
            if first_occ[pos] is not None and first_occ[pos + 1] is not None:
                a = first_occ[pos]
                b = first_occ[pos + 1]
                if valid[a] and valid[b]:
                    r_a = couples[a][1]
                    r_b = couples[b][1]
                    if abs(r_a - r_b) == 1:
                        ans += 1
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()