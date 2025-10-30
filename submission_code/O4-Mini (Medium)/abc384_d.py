import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    S = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Total sum of one period
    T = sum(A)
    # Build B = A + A for two periods
    B = A + A
    # Prefix sums P[0..2N]
    P = [0] * (2 * N + 1)
    for i in range(2 * N):
        P[i+1] = P[i] + B[i]
    # Target residue modulo T
    m = S % T
    # We'll maintain a sliding window of l-indices in [r-N .. r-1],
    # storing counts of P[l] values in a dict.
    from collections import Counter
    M = Counter()
    # Iterate r from 1 to 2N
    # For each r, before checking, we add l = r-1 into M,
    # and remove l = r-N-1 if it falls out.
    # Then we want to know if exists l in M such that
    #   P[r] - P[l] ≡ S (mod T)
    # and
    #   P[r] - P[l] <= S
    # The modulo condition is: P[l] ≡ P[r] - m  (mod T)
    # Since P[l] ∈ [0..2T], two possibilities for P[l]:
    #   q = (P[r] - m) % T, or q+T.
    # And we also require P[l] >= P[r] - S.
    for r in range(1, 2 * N + 1):
        # add l = r-1
        l_add = r - 1
        M[P[l_add]] += 1
        # remove l = r-N-1 if >= 0
        l_rem = r - N - 1
        if l_rem >= 0:
            pv = P[l_rem]
            cnt = M[pv]
            if cnt == 1:
                del M[pv]
            else:
                M[pv] = cnt - 1
        # compute needed residue q
        # want P[l] ≡ P[r] - m  (mod T)
        pr = P[r]
        q = (pr - m) % T
        min_pl = pr - S  # need P[l] >= min_pl
        # check P[l] == q
        if q in M:
            # check if q >= min_pl
            if q >= min_pl:
                print("Yes")
                return
        # check P[l] == q + T
        qT = q + T
        if qT in M:
            if qT >= min_pl:
                print("Yes")
                return
    # if none found
    print("No")

if __name__ == "__main__":
    main()