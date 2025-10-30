import sys
def main():
    input = sys.stdin.readline
    N, A, B = map(int, input().split())
    Ds = list(map(int, input().split()))
    L = A + B

    # curr holds the current allowed y‐residue intervals (0<=y<L),
    # such that for all processed D_i, (y + D_i) mod L is in [0..A-1].
    curr = [(0, L - 1)]

    for D in Ds:
        # Compute the allowed interval(s) U for this D:
        # we need (y + D) mod L in [0..A-1]
        # <=> y mod L in [-D .. -D + (A-1)] mod L
        l = (-D) % L
        r = l + (A - 1)
        if r < L:
            segs = [(l, r)]
        else:
            # wraps around
            segs = [(l, L - 1), (0, r % L)]

        # intersect curr with segs
        new_curr = []
        for (a, b) in curr:
            for (c, d) in segs:
                lo = a if a > c else c
                hi = b if b < d else d
                if lo <= hi:
                    new_curr.append((lo, hi))

        if not new_curr:
            print("No")
            return

        # merge overlapping intervals in new_curr
        if len(new_curr) > 1:
            new_curr.sort()
            merged = []
            s, e = new_curr[0]
            for (ss, ee) in new_curr[1:]:
                if ss <= e + 1:
                    # overlap or touch
                    if ee > e:
                        e = ee
                else:
                    merged.append((s, e))
                    s, e = ss, ee
            merged.append((s, e))
            curr = merged
        else:
            curr = new_curr

    # If we still have any allowed residue, answer is Yes
    print("Yes")


if __name__ == "__main__":
    main()