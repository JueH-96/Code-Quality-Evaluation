import sys
import threading
def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    C = [int(next(it)) for _ in range(N)]

    # Initial weight
    W0 = 0
    for i in range(N):
        if A[i] == 1:
            W0 += C[i]

    # Classify bits
    S_minus = []   # A_i=1, B_i=0: need delta -C
    S_plus = []    # A_i=0, B_i=1: need delta +C
    stable = []    # A_i=1, B_i=1: candidate extra flips
    for i in range(N):
        ai = A[i]; bi = B[i]; ci = C[i]
        if ai == 1 and bi == 0:
            S_minus.append(ci)
        elif ai == 0 and bi == 1:
            S_plus.append(ci)
        elif ai == 1 and bi == 1:
            stable.append(ci)
    # Prepare all delta values for BIT coordinate compression
    # We need both +c and -c for c in S_minus, S_plus, stable
    vals = set()
    for c in S_minus:
        vals.add(-c)
        # c itself maybe already in S_plus or stable; add anyway
    for c in S_plus:
        vals.add(c)
    for c in stable:
        vals.add(c)
        vals.add(-c)
    # Build sorted list
    val_list = sorted(vals)
    mval = {v:i for i,v in enumerate(val_list)}
    size = len(val_list)

    # Fenwick/BIT arrays
    bitC = [0] * size  # counts
    bitS = [0] * size  # sums of delta values
    # counts_map to rebuild final D
    counts_map = [0] * size

    # Helper: update BIT at i by delta_count in bitC and delta_sum in bitS
    def bit_update(i, dc, ds):
        # update count tree
        n = size
        # count
        j = i
        while j < n:
            bitC[j] += dc
            bitS[j] += ds
            j |= j + 1

    # Helper: prefix sum query of bitC and bitS up to idx inclusive
    def bit_prefix(idx):
        # returns (count_sum, value_sum)
        csum = 0; vsum = 0
        j = idx
        while j >= 0:
            csum += bitC[j]
            vsum += bitS[j]
            j = (j & (j + 1)) - 1
        return csum, vsum

    # Build initial D0 from mismatches
    M = 0
    for c in S_minus:
        idx = mval[-c]
        counts_map[idx] += 1
        # one count update, sum update by -c
        bit_update(idx, 1, -c)
        M += 1
    for c in S_plus:
        idx = mval[c]
        counts_map[idx] += 1
        bit_update(idx, 1, c)
        M += 1

    # Greedy include stable bits sorted descending by C
    if stable:
        stable.sort(reverse=True)
        # Pre-bind locals for speed
        _bit_prefix = bit_prefix
        _bit_update = bit_update
        _mval = mval
        _size = size
        _counts_map = counts_map
        _W0 = W0
        _bitC = bitC; _bitS = bitS
        for c in stable:
            # compute count/sum of d < -c
            idxn = _mval.get(-c)
            if idxn is None:
                # if -c not in mapping, then no smaller elements, treat as 0
                cnt_neg = 0; sum_neg = 0
                # but for insertion we still need map -c
                # Actually all stable c added to mapping
                # Should not happen
            else:
                if idxn - 1 >= 0:
                    cnt_neg, sum_neg = _bit_prefix(idxn - 1)
                else:
                    cnt_neg = 0; sum_neg = 0
            # compute count/sum of d < +c
            idxp = _mval[c]  # c always in mapping
            if idxp - 1 >= 0:
                cnt_pos, sum_pos = _bit_prefix(idxp - 1)
            else:
                cnt_pos = 0; sum_pos = 0

            # s1 = sum of d < -c
            s1 = sum_neg
            # s2 = sum of d < +c, but D0 has no +c yet; removal insertion at D1 adds -c then s2 sums D1
            # we derived s2 = sum_old_less_c - c
            s2 = sum_pos - c

            # compute DeltaF and DeltaG
            # DeltaF = s1 + s2 + (cnt_neg - cnt_pos)*c
            df = s1 + s2 + (cnt_neg - cnt_pos) * c
            deltaG = 2 * _W0 + df
            if deltaG < 0:
                # include stable bit: insert -c and +c
                # update mapping
                # insert -c
                idxn = _mval[-c]
                _counts_map[idxn] += 1
                _bit_update(idxn, 1, -c)
                # insert +c
                idxp = _mval[c]
                _counts_map[idxp] += 1
                _bit_update(idxp, 1, c)
                # increase M by 2
                M += 2
            else:
                break

    # Now rebuild sorted delta list D_final from counts_map
    # We'll iterate over val_list (sorted) and repeat each value counts_map times
    D_final = []
    # Reserve list capacity? Not needed
    for idx, val in enumerate(val_list):
        cnt = counts_map[idx]
        if cnt:
            # extend by repeating val cnt times
            D_final += [val] * cnt

    # Now simulate cost: start with currW = W0, for each delta apply and accumulate
    currW = W0
    ans = 0
    for d in D_final:
        currW += d
        ans += currW

    # Output result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()