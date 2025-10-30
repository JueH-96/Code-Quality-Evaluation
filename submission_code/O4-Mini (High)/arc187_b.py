import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    B = [0]*n
    for i in range(n):
        b = int(next(it))
        B[i] = b  # -1 or 1..m

    mod = 998244353
    # Precompute suffix differences: suf_diff[j] for j in [0..n-1]
    # suf_diff[j][s] = ways suffix [j+1..n-1] has max exactly s.
    from array import array
    # suffix_le[s] = ways assign suffix positions kj+1..n-1 such that all <= s
    # start with suffix_le for empty suffix (k=n): suffix_le[s]=1 for all s>=0
    suffix_le = [1] * (m+1)
    # allocate list for diffs
    suf_diff = [None] * n
    # build from bottom
    for k in range(n-1, -1, -1):
        # compute diff for position k
        row = [0] * (m+1)
        prev = 0
        # for s from 0..m
        # row[s] = (suffix_le[s] - prev) mod
        for s in range(m+1):
            curr = suffix_le[s]
            d = curr - prev
            if d < 0:
                d += mod
            row[s] = d
            prev = curr
        # store as unsigned array
        suf_diff[k] = array('I', row)
        # update suffix_le to include B[k] at front (position k)
        bk = B[k]
        if bk == -1:
            # wildcard: choices for <= s is s
            # for s=0: no choices ->0
            suffix_le[0] = 0
            # for s>=1: multiply by s
            # local var for speed
            mul = 1
            for s in range(1, m+1):
                # suffix_le[s] = suffix_le[s] * s % mod
                # use s as multiplier
                v = suffix_le[s] * s
                # reduce mod
                v %= mod
                suffix_le[s] = v
        else:
            x = bk
            # for s<x: no assignment ->0; for s>=x: keep
            # set suffix_le[0..x-1]=0
            # local ref
            # if x > 0:
            for s in range(x):
                suffix_le[s] = 0
            # s >= x: suffix_le[s] stays
            # nothing to do

    # Now prefix pass
    ans = 0
    # prefix_gt[T] = ways assign prefix [0..j-1] values > T
    # initialize for j=0 prefix empty
    prefix_gt = [1] * (m+1)
    # iterate j from 0..n-1
    for j in range(n):
        pref = prefix_gt  # length m+1
        diffj = suf_diff[j]  # array('I')
        bj = B[j]
        if bj == -1:
            # wildcard at j: need full formula
            # compute sufP: suffix sums of pref from right
            # sufP[s] = sum_{t=s..m} pref[t] mod
            sufP = [0] * (m+2)
            cum = 0
            # s=m..0
            for s in range(m, -1, -1):
                cum += pref[s]
                if cum >= mod:
                    cum -= mod
                sufP[s] = cum
            # compute sum_Cj
            total = 0
            # local refs
            modloc = mod
            pref_loc = pref
            sufP_loc = sufP
            diff_loc = diffj
            # loop s=0..m
            for s in range(m+1):
                # F = (s-1)*pref[s] + sufP[s] mod
                # compute t = (s-1)*pref_loc[s] %modloc
                # note s-1 may be -1 when s=0
                t = (s-1) * pref_loc[s] + sufP_loc[s]
                # reduce mod
                t %= modloc
                # add diff_loc[s] * t
                total += diff_loc[s] * t
                # reduce occasionally
                if total >= 1<<63:
                    total %= modloc
            sum_Cj = total % mod
        else:
            # fixed bj
            x = bj
            # accumulate suf_le_cum and T1_cum
            suf_le_cum = 0
            T1_cum = 0
            modloc = mod
            pref_loc = pref
            diff_loc = diffj
            f0 = 0
            t0 = 0
            # s from 0..m
            for s in range(m+1):
                ds = diff_loc[s]
                # update suf_le_cum
                suf_le_cum += ds
                if suf_le_cum >= modloc:
                    suf_le_cum -= modloc
                # update T1_cum
                T1_cum += ds * pref_loc[s]
                if T1_cum >= modloc:
                    T1_cum %= modloc
                # capture at s == x
                if s == x:
                    f0 = suf_le_cum
                    t0 = T1_cum
            # T1_total = T1_cum
            # sum_Cj = f0*pref[x] + (T1_total - t0)
            sum_Cj = f0 * pref_loc[x] + (T1_cum - t0)
            sum_Cj %= modloc
        # add to ans
        ans += sum_Cj
        if ans >= mod:
            ans %= mod

        # update prefix_gt for next j+1
        bk = bj
        if bk == -1:
            # wildcard: multiply by (m - T)
            # prefix_gt[T] = prefix_gt[T] * (m - T) % mod
            modloc = mod
            # local ref for speed
            # compute for T=0..m
            for s in range(m+1):
                # compute factor = m - s
                # prefix_gt[s] = prefix_gt[s] * factor %modloc
                v = prefix_gt[s] * (m - s)
                if v >= modloc:
                    v %= modloc
                prefix_gt[s] = v
        else:
            x = bk
            # for T >= x: prefix_gt[T] = 0
            # for T<x: prefix_gt[T] remains
            # so zero out entries x..m
            # local ref
            for s in range(x, m+1):
                prefix_gt[s] = 0

    # final answer
    ans %= mod
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()