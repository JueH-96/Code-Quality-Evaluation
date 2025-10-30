import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N-1)]
    mod = 998244353
    M = N - 1
    # compute smallest prime factor up to max(A)
    maxA = max(A) if A else 1
    spf = [0] * (maxA + 1)
    for i in range(2, maxA+1):
        if spf[i] == 0:
            for j in range(i, maxA+1, i):
                if spf[j] == 0:
                    spf[j] = i
    # factor each A[i]
    A_exps = [dict() for _ in range(M)]
    primes = set()
    for idx, v in enumerate(A):
        x = v
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            A_exps[idx][p] = cnt
            primes.add(p)
    # build exponent lists for each prime
    prime_exp = {}
    for p in primes:
        el = [0] * M
        for i in range(M):
            el[i] = A_exps[i].get(p, 0)
        prime_exp[p] = el
    ans = 1
    # for each prime p, do DP
    for p, exp_list in prime_exp.items():
        # sum of exponents
        sum_a = 0
        for e in exp_list:
            sum_a += e
        # handle N=2 (M=1)
        if M == 1:
            a1 = exp_list[0]
            if a1 > 0:
                # two sequences, each weight p^a1
                ans = ans * (2 * pow(p, a1, mod) % mod) % mod
            # if a1==0, contributes 1
            continue
        # M >= 2
        # precompute p^k up to 2*sum_a + a_M
        aM = exp_list[M-1]
        max_pow = 2 * sum_a + aM
        p_pows = [1] * (max_pow + 1)
        for i in range(1, max_pow+1):
            p_pows[i] = p_pows[i-1] * p % mod
        # DP arrays, dynamic length
        # dp_total_prev0,1 and dp_false_prev0,1
        # initialize j=1
        a1 = exp_list[0]
        if a1 > 0:
            prev_max_y = a1
            dp_tot_prev0 = [1]           # y1=0, d1=0
            dp_false_prev0 = [0]
            # for d1 = a1
            lst0 = dp_tot_prev0
            dp_tot_prev1 = [0] * (a1) + [p_pows[a1]]
            dp_false_prev1 = [0] * (a1) + [p_pows[a1]]
        else:
            prev_max_y = 0
            dp_tot_prev0 = [1]
            dp_false_prev0 = [0]
            dp_tot_prev1 = []
            dp_false_prev1 = []
        # DP for j = 2..M-1
        for idx in range(1, M-1):
            a_prev = exp_list[idx-1]
            a_curr = exp_list[idx]
            new_max_y = prev_max_y + a_prev
            # prepare new DP arrays
            # length new_max_y+1
            tot_new0 = [0] * (new_max_y + 1)
            tot_new1 = [0] * (new_max_y + 1)
            fal_new0 = [0] * (new_max_y + 1)
            fal_new1 = [0] * (new_max_y + 1)
            # local refs
            prev0 = dp_tot_prev0; prev1 = dp_tot_prev1
            fprev0 = dp_false_prev0; fprev1 = dp_false_prev1
            pows = p_pows; md = mod
            # flag=0 transitions: d_prev=0
            # y_j = y_prev + a_prev
            delta = a_prev
            # iterate y_prev from 0..prev_max_y
            # prev0 and fprev0 have length prev_max_y+1
            for y_prev in range(prev_max_y+1):
                w_tot = prev0[y_prev]
                w_f = fprev0[y_prev]
                if w_tot == 0 and w_f == 0:
                    continue
                yj = y_prev + delta
                # yj >= 0 always
                wy = pows[yj]
                # d_j = 0 case
                if w_tot:
                    tot_new0[yj] = (tot_new0[yj] + w_tot * wy) % md
                if w_f and yj >= 1:
                    fal_new0[yj] = (fal_new0[yj] + w_f * wy) % md
                # d_j = a_curr case
                if a_curr and yj >= a_curr:
                    if w_tot:
                        tot_new1[yj] = (tot_new1[yj] + w_tot * wy) % md
                    if w_f and yj >= 1:
                        fal_new1[yj] = (fal_new1[yj] + w_f * wy) % md
            # flag=1 transitions: d_prev = a_prev, only if a_prev>0
            if a_prev:
                delta = -a_prev
                # prev1, fprev1 length = prev_max_y+1 (or 0 if empty)
                # y_prev must be >= a_prev for yj>=0
                # so y_prev in [a_prev..prev_max_y]
                for y_prev in range(a_prev, prev_max_y+1):
                    w_tot = prev1[y_prev] if y_prev < len(prev1) else 0
                    w_f = fprev1[y_prev] if y_prev < len(fprev1) else 0
                    if w_tot == 0 and w_f == 0:
                        continue
                    yj = y_prev + delta
                    # yj >=0 by loop bounds
                    wy = pows[yj]
                    # d_j = 0
                    if w_tot:
                        tot_new0[yj] = (tot_new0[yj] + w_tot * wy) % md
                    if w_f and yj >= 1:
                        fal_new0[yj] = (fal_new0[yj] + w_f * wy) % md
                    # d_j = a_curr
                    if a_curr and yj >= a_curr:
                        if w_tot:
                            tot_new1[yj] = (tot_new1[yj] + w_tot * wy) % md
                        if w_f and yj >= 1:
                            fal_new1[yj] = (fal_new1[yj] + w_f * wy) % md
            # swap dp arrays
            dp_tot_prev0 = tot_new0
            dp_tot_prev1 = tot_new1
            dp_false_prev0 = fal_new0
            dp_false_prev1 = fal_new1
            prev_max_y = new_max_y
        # final stage j=M
        a_prev = exp_list[M-2]
        a_curr = exp_list[M-1]
        # accumulate S_total and S_fail
        S_tot = 0
        S_fal = 0
        pows = p_pows; md = mod
        prev0 = dp_tot_prev0; prev1 = dp_tot_prev1
        fprev0 = dp_false_prev0; fprev1 = dp_false_prev1
        # flag=0
        delta = a_prev
        # y_prev in 0..prev_max_y
        for y_prev in range(prev_max_y+1):
            w_tot = prev0[y_prev]
            w_f = fprev0[y_prev]
            if w_tot == 0 and w_f == 0:
                continue
            yM = y_prev + delta
            if yM < 0:
                continue
            # dM = 0
            xp = yM
            idxp = 2*xp + a_curr
            mul = pows[idxp]
            if w_tot:
                S_tot = (S_tot + w_tot * mul) % md
            if w_f and yM >= 1:
                S_fal = (S_fal + w_f * mul) % md
            # dM = a_curr
            if a_curr and yM >= a_curr:
                xp1 = yM - a_curr
                idx1 = 2*xp1 + a_curr
                mul1 = pows[idx1]
                if w_tot:
                    S_tot = (S_tot + w_tot * mul1) % md
                # fail requires yM > a_curr
                if w_f and yM > a_curr:
                    S_fal = (S_fal + w_f * mul1) % md
        # flag=1
        if a_prev:
            delta = -a_prev
            # y_prev in a_prev..prev_max_y
            # prev1 and fprev1 have same length prev_max_y+1
            for y_prev in range(a_prev, prev_max_y+1):
                w_tot = prev1[y_prev] if y_prev < len(prev1) else 0
                w_f = fprev1[y_prev] if y_prev < len(fprev1) else 0
                if w_tot == 0 and w_f == 0:
                    continue
                yM = y_prev + delta
                if yM < 0:
                    continue
                xp = yM
                idxp = 2*xp + a_curr
                mul = pows[idxp]
                if w_tot:
                    S_tot = (S_tot + w_tot * mul) % md
                if w_f and yM >= 1:
                    S_fal = (S_fal + w_f * mul) % md
                if a_curr and yM >= a_curr:
                    xp1 = yM - a_curr
                    idx1 = 2*xp1 + a_curr
                    mul1 = pows[idx1]
                    if w_tot:
                        S_tot = (S_tot + w_tot * mul1) % md
                    if w_f and yM > a_curr:
                        S_fal = (S_fal + w_f * mul1) % md
        Sp = (S_tot - S_fal) % md
        ans = ans * Sp % mod
    # print result
    print(ans)

if __name__ == "__main__":
    main()