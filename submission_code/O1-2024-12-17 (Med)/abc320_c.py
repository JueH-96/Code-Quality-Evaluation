def main():
    import sys
    data = sys.stdin.read().strip().split()
    M = int(data[0])
    S = data[1:]  # S[0], S[1], S[2] are the reels

    # positions[i][d] will hold all indices r where S[i][r] == str(d)
    positions = [[[] for _ in range(10)] for _ in range(3)]
    for i in range(3):
        for r in range(M):
            digit = int(S[i][r])
            positions[i][digit].append(r)

    from math import inf
    ans = inf

    # Precompute permutations of reels.
    perms = [(0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0)]

    # Function to find the next time strictly greater than curr
    # whose remainder mod M is target_mod.
    def next_distinct_time(curr, target_mod, M):
        r = curr % M
        t = curr - r + target_mod
        if t <= curr:
            t += M
        return t

    # Try each possible digit d from 0..9 for alignment.
    for d in range(10):
        # If any reel does not contain this digit, skip.
        if not positions[0][d] or not positions[1][d] or not positions[2][d]:
            continue

        # For each permutation of (0,1,2), representing the order
        # we press the reels.
        for p1, p2, p3 in perms:
            # Precompute for the third reel (p3) a table of minimal steps
            # to get from remainder r to some remainder in positions[p3][d].
            # next_rem[r] = minimal positive k such that (r + k) % M is in positions[p3][d].
            # If (r + k) % M == r for some x3 == r, we use k = M to ensure the new time is > current.
            R3 = positions[p3][d]
            next_rem = [inf] * M
            for r in range(M):
                best_diff = inf
                for x3 in R3:
                    diff = x3 - r
                    if diff < 0:
                        diff += M
                    # If diff == 0, that would mean the same time,
                    # so we use diff = M to ensure distinct time.
                    if diff == 0:
                        diff = M
                    if diff < best_diff:
                        best_diff = diff
                next_rem[r] = best_diff

            # Now press reels p1, p2, then p3 in that order.
            A1 = positions[p1][d]
            A2 = positions[p2][d]
            for x1 in A1:
                # Earliest time to press reel p1 that yields remainder x1 is just x1.
                a = x1
                for x2 in A2:
                    # Next time strictly greater than a that has remainder x2:
                    b = next_distinct_time(a, x2, M)
                    rb = b % M
                    diff_c = next_rem[rb]
                    if diff_c == inf:
                        continue
                    c = b + diff_c
                    if c < ans:
                        ans = c

    print(ans if ans < float('inf') else -1)

# Don't forget to call main()
if __name__ == "__main__":
    main()