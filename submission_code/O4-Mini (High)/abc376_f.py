def main():
    import sys
    input = sys.stdin.readline
    INF = 10**18

    # Read input
    line = input().split()
    N = int(line[0]); Q = int(line[1])
    H = [None]*Q
    T = [0]*Q
    for i in range(Q):
        hi_ti = input().split()
        H[i] = hi_ti[0]
        T[i] = int(hi_ti[1])

    # Prepare dp arrays (size N+1, 1-based)
    Np1 = N + 1
    dp0 = [INF] * Np1
    dp1 = [0] * Np1
    dp_old = dp0
    dp_new = dp1

    # Initialize dp_old for the first instruction
    H0 = H[0]
    t0 = T[0]
    if H0 == 'L':
        # Left moves from 1 -> t0, Right from 2 -> b
        d = 1 - t0
        if d < 0: d = -d
        if d > N - d: dL = N - d
        else: dL = d
        dol = dp_old
        for b in range(1, Np1):
            if b == t0:
                dol[b] = INF
            else:
                diff = 2 - b
                if diff < 0: diff = -diff
                if diff > N - diff: dR = N - diff
                else: dR = diff
                dol[b] = dL + dR
    else:
        # Right moves from 2 -> t0, Left from 1 -> b
        d = 2 - t0
        if d < 0: d = -d
        if d > N - d: dR = N - d
        else: dR = d
        dol = dp_old
        for b in range(1, Np1):
            if b == t0:
                dol[b] = INF
            else:
                diff = 1 - b
                if diff < 0: diff = -diff
                if diff > N - diff: dL = N - diff
                else: dL = diff
                dol[b] = dR + dL

    # Buffers for convolution (reuse each time)
    left  = [0] * Np1
    m2_1  = [0] * Np1
    right = [0] * Np1
    m2_2  = [0] * Np1

    prev_H = H0
    prev_T = t0

    # Process the remaining instructions
    for i in range(1, Q):
        Hi = H[i]
        Ti = T[i]
        dp_o = dp_old
        dp_n = dp_new

        if Hi == prev_H:
            # Case: same hand moves again -> convolution
            diff = prev_T - Ti
            if diff < 0: diff = -diff
            if diff > N - diff: D = N - diff
            else: D = diff

            # Exclude the position prev_T
            dp_o[prev_T] = INF

            # Forward pass: compute left[b] and m2_1[b]
            f1 = INF
            pref = INF
            for b in range(1, Np1):
                v = dp_o[b]
                t1 = v - b
                if t1 < f1: f1 = t1
                left[b] = f1 + b
                t2 = v + b
                if t2 < pref: pref = t2
                m2_1[b] = pref - b

            # Backward pass: compute right[b] and m2_2[b]
            f2 = INF
            suff = INF
            for b in range(N, 0, -1):
                v = dp_o[b]
                t3 = v + b
                if t3 < f2: f2 = t3
                right[b] = f2 - b
                t4 = v - b
                if t4 < suff: suff = t4
                m2_2[b] = suff + b

            # Build dp_new
            for b in range(1, Np1):
                if b == Ti:
                    dp_n[b] = INF
                else:
                    # M1 = min(left[b], right[b])
                    x = left[b]; y = right[b]
                    if y < x: x = y
                    # M2 = N + min(m2_1[b], m2_2[b])
                    z = m2_1[b]; w = m2_2[b]
                    if w < z: z = w
                    # minimal over the two directions
                    mval = x if x < z + N else (z + N)
                    dp_n[b] = D + mval

        else:
            # Case: hand switches -> simple scan for C
            dp_o[prev_T] = INF
            C = INF
            ti = Ti
            NN = N
            for a in range(1, Np1):
                v = dp_o[a]
                if v < C:
                    diff = a - ti
                    if diff < 0: diff = -diff
                    if diff > NN - diff: d = NN - diff
                    else: d = diff
                    tmp = v + d
                    if tmp < C: C = tmp
            pT = prev_T
            for b in range(1, Np1):
                if b == Ti:
                    dp_n[b] = INF
                else:
                    diff = pT - b
                    if diff < 0: diff = -diff
                    if diff > N - diff: d2 = N - diff
                    else: d2 = diff
                    dp_n[b] = C + d2

        # Swap dp_old and dp_new, update prev values
        prev_H = Hi
        prev_T = Ti
        dp_old, dp_new = dp_new, dp_old

    # The answer is the minimal dp_old[b] (b != T_Q, but dp_old[T_Q] is INF anyway)
    ans = min(dp_old[1:])  # skip index 0
    print(ans)

if __name__ == "__main__":
    main()