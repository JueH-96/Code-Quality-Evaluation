# YOUR CODE HERE
import sys

import threading

def main():
    import sys

    import math

    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    R = set()
    C = set()
    Dp = set()
    Dn = set()
    existing_pieces = set()
    for _ in range(M):
        a_k, b_k = map(int, sys.stdin.readline().split())
        R.add(a_k)
        C.add(b_k)
        Dp.add(a_k + b_k)
        Dn.add(a_k - b_k)
        existing_pieces.add((a_k, b_k))

    N_R = len(R)
    N_C = len(C)
    Total_R = N_R * N
    Total_C = N_C * N

    # Compute lengths of positive diagonals
    Total_Dp = 0
    for s in Dp:
        # Diagonals from (i,j) where i + j = s
        # The range of i and j satisfying 1 <= i,j <= N
        min_i = max(1, s - N)
        max_i = min(N, s - 1)
        length = max(0, max_i - min_i + 1)
        Total_Dp += length

    # Compute lengths of negative diagonals
    Total_Dn = 0
    for d in Dn:
        # Diagonals where i - j = d
        min_i = max(1, d + 1)
        max_i = min(N, N + d)
        length = max(0, max_i - min_i + 1)
        Total_Dn += length

    # Pairwise overlaps
    Overlap_RC = N_R * N_C

    Overlap_R_Dp = 0
    for r in R:
        for s in Dp:
            c = s - r
            if 1 <= c <= N:
                Overlap_R_Dp += 1

    Overlap_R_Dn = 0
    for r in R:
        for d in Dn:
            c = r - d
            if 1 <= c <= N:
                Overlap_R_Dn += 1

    Overlap_C_Dp = 0
    for c in C:
        for s in Dp:
            r = s - c
            if 1 <= r <= N:
                Overlap_C_Dp += 1

    Overlap_C_Dn = 0
    for c in C:
        for d in Dn:
            r = c + d
            if 1 <= r <= N:
                Overlap_C_Dn += 1

    Overlap_Dp_Dn = 0
    for s in Dp:
        for d in Dn:
            if (s + d) % 2 == 0:
                r = (s + d) // 2
                c = (s - d) // 2
                if 1 <= r <= N and 1 <= c <= N:
                    Overlap_Dp_Dn += 1

    # Triple overlaps
    Overlap_R_C_Dp = 0
    for r in R:
        for c in C:
            s = r + c
            if s in Dp:
                Overlap_R_C_Dp += 1

    Overlap_R_C_Dn = 0
    for r in R:
        for c in C:
            d = r - c
            if d in Dn:
                Overlap_R_C_Dn += 1

    Overlap_R_Dp_Dn = 0
    for r in R:
        for s in Dp:
            d = r - (s - r)
            if d in Dn:
                c = s - r
                if 1 <= c <= N:
                    Overlap_R_Dp_Dn += 1

    Overlap_C_Dp_Dn = 0
    for c in C:
        for s in Dp:
            d = (s - c) - c
            if d in Dn:
                r = s - c
                if 1 <= r <= N:
                    Overlap_C_Dp_Dn += 1

    # Quadruple overlaps
    Overlap_R_C_Dp_Dn = 0
    for r in R:
        for c in C:
            s = r + c
            d = r - c
            if s in Dp and d in Dn:
                Overlap_R_C_Dp_Dn += 1

    Total_Pairwise_Overlaps = Overlap_RC + Overlap_R_Dp + Overlap_R_Dn + Overlap_C_Dp + Overlap_C_Dn + Overlap_Dp_Dn
    Total_Triple_Overlaps = Overlap_R_C_Dp + Overlap_R_C_Dn + Overlap_R_Dp_Dn + Overlap_C_Dp_Dn
    Total_Quadruple_Overlaps = Overlap_R_C_Dp_Dn

    Net_Attacked_Cells = Total_R + Total_C + Total_Dp + Total_Dn \
                       - Total_Pairwise_Overlaps \
                       + Total_Triple_Overlaps \
                       - Total_Quadruple_Overlaps

    # Since the cells occupied by the existing pieces have been counted in the attack counts,
    # we need to adjust for them.

    Answer = N * N - Net_Attacked_Cells

    print(Answer)

threading.Thread(target=main).start()