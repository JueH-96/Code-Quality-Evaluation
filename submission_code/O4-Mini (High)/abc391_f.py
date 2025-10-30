#!/usr/bin/env python3
import sys

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))

    # Read and sort the three arrays in descending order
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    C = [int(next(it)) for _ in range(N)]
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # We will use a max‐heap over (i,j,k) triples, but Python only has a min‐heap,
    # so we push negative values.  We also need to mark visited (i,j,k) so we
    # don't revisit the same triple.  We flatten (i,j,k) into a single integer
    # code = (i<<2w) | (j<<w) | k, where w is the bit‐width to hold N-1.
    SHIFT1 = (N-1).bit_length()
    SHIFT2 = SHIFT1 * 2

    from heapq import heappush, heappop

    visited = set()
    vadd = visited.add
    vcontains = visited.__contains__

    # Local aliases for speed
    A_l, B_l, C_l = A, B, C
    N_l = N
    hpush, hpop = heappush, heappop

    # Priority queue: entries are (–value, i, j, k)
    pq = []

    # Seed with (0,0,0)
    ai0 = A_l[0]; bj0 = B_l[0]; ck0 = C_l[0]
    v0 = ai0*bj0 + bj0*ck0 + ck0*ai0
    hpush(pq, (-v0, 0, 0, 0))
    vadd(0)  # code for (0,0,0) is 0

    answer = 0
    # Extract the top K triples
    for _ in range(K):
        negv, i, j, k = hpop(pq)
        answer = -negv

        # Try pushing the three neighbors if not yet visited:
        # (i+1, j, k), (i, j+1, k), (i, j, k+1)
        ni = i + 1
        if ni < N_l:
            code = (ni << SHIFT2) | (j << SHIFT1) | k
            if not vcontains(code):
                vadd(code)
                ai1 = A_l[ni]; bj1 = B_l[j]; ck1 = C_l[k]
                vv = ai1*bj1 + bj1*ck1 + ck1*ai1
                hpush(pq, (-vv, ni, j, k))

        nj = j + 1
        if nj < N_l:
            code = (i << SHIFT2) | (nj << SHIFT1) | k
            if not vcontains(code):
                vadd(code)
                ai1 = A_l[i]; bj1 = B_l[nj]; ck1 = C_l[k]
                vv = ai1*bj1 + bj1*ck1 + ck1*ai1
                hpush(pq, (-vv, i, nj, k))

        nk = k + 1
        if nk < N_l:
            code = (i << SHIFT2) | (j << SHIFT1) | nk
            if not vcontains(code):
                vadd(code)
                ai1 = A_l[i]; bj1 = B_l[j]; ck1 = C_l[nk]
                vv = ai1*bj1 + bj1*ck1 + ck1*ai1
                hpush(pq, (-vv, i, j, nk))

    # Print the K-th largest value
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()