import sys
import threading

def main():
    import sys
    from itertools import permutations

    input = sys.stdin.readline
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]

    # Count inversions in a permutation p of length n.
    def inv_count(p):
        cnt = 0
        n = len(p)
        for i in range(n):
            for j in range(i+1, n):
                if p[i] > p[j]:
                    cnt += 1
        return cnt

    best = None

    rows = range(H)
    cols = range(W)
    # Try every way to permute the rows of A (pi)
    for pi in permutations(rows):
        # Try every way to permute the columns of A (sigma)
        for sigma in permutations(cols):
            ok = True
            for i in range(H):
                ai = pi[i]
                # we know row i in the target B must come from row ai in A
                # and similarly column j in B must come from column sigma[j] in A
                # so we need A[ai][ sigma[j] ] == B[i][j] for all j
                rowA = A[ai]
                rowB = B[i]
                for j in range(W):
                    if rowA[sigma[j]] != rowB[j]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # if we get here, this (pi, sigma) makes A identical to B
            cost = inv_count(pi) + inv_count(sigma)
            if best is None or cost < best:
                best = cost

    print(best if best is not None else -1)

if __name__ == "__main__":
    main()