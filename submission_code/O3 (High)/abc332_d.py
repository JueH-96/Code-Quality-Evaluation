import sys
import itertools

def inversion_count(perm):
    """Return the number of inversions in the permutation `perm`."""
    cnt = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                cnt += 1
    return cnt


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))

    A = [[int(next(it)) for _ in range(W)] for _ in range(H)]
    B = [[int(next(it)) for _ in range(W)] for _ in range(H)]

    # Pre-compute all permutations of rows and columns together with
    # their inversion counts so that we do not recompute them repeatedly.
    row_perms = []
    for p in itertools.permutations(range(H)):
        row_perms.append((p, inversion_count(p)))

    col_perms = []
    for q in itertools.permutations(range(W)):
        col_perms.append((q, inversion_count(q)))

    INF = 10 ** 18
    best = INF

    # Try every pair of (row permutation, column permutation)
    for p, inv_p in row_perms:
        # We can take a small shortcut: build the row-permuted version once
        # to speed up the inner loop over column permutations.
        permuted_rows = [A[p_i] for p_i in p]  # list of lists (rows)

        for q, inv_q in col_perms:
            ok = True
            # Check equality cell by cell; break early on mismatch.
            for i in range(H):
                row_p = permuted_rows[i]
                row_b = B[i]
                for j in range(W):
                    if row_p[q[j]] != row_b[j]:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                best = min(best, inv_p + inv_q)

    print(-1 if best == INF else best)


if __name__ == "__main__":
    main()