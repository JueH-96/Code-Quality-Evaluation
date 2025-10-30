def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    S = input_data[2]
    mod = 998244353

    # -----------------------------------------------
    # 1) Precompute possible toggles for each position.
    #    We'll represent parity states as integers 0..7 (bits for A,B,C).
    #    T(A) = 1 (toggle bit0), T(B) = 2 (bit1), T(C) = 4 (bit2).
    # -----------------------------------------------
    def letter_to_toggle(ch):
        if ch == 'A': return [1]
        if ch == 'B': return [2]
        if ch == 'C': return [4]
        # ch == '?'
        return [1,2,4]

    toggles = []
    for ch in S:
        toggles.append(letter_to_toggle(ch))

    # -----------------------------------------------
    # 2) We will encode a "parity state" r in [0..7].
    #    p(i) = r means the XOR of bits up to i is r.
    #    We also need to track how many times each parity has occurred
    #    in order to compute how many new good substrings each step adds.
    #
    #    However, a direct freq-array-of-8 approach is too large to store.
    #    Instead, note that to count new good substrings at step i when
    #    p(i)=r, we need freq(r)+ freq(r^7). But freq(r)+freq(r^7) = x[p],
    #    where we pair up r with r^7. We keep four "pairs" of parity states:
    #       pair(0)=0<->7, pair(1)=1<->6, pair(2)=2<->5, pair(3)=3<->4.
    #    Let pairIndex(r) be which of the 4 pairs r belongs to,
    #    and signBit(r) = +1 if r is the smaller in that pair, -1 if bigger.
    #
    #    We store x0,x1,x2,x3 as how many times each pair is used total,
    #    and d0,d1,d2,d3 as the "difference" so that freq(r) can be derived
    #    if needed. But for counting new substrings formed by picking p(i)=r,
    #    we only need x_{pairIndex(r)} (before increment). That is exactly
    #    freq(r)+freq(r^7).
    #
    #    Transition:
    #       old state: (r, x0,x1,x2,x3, d0,d1,d2,d3)
    #       pick toggle t => r' = r ^ t
    #       new good substrings = x_{ pairIndex(r') }
    #       then x'_{pIndex(r')}= x_{pIndex(r')}+1
    #            d'_{pIndex(r')}= d_{pIndex(r')} + signBit(r')
    #       store in dp[i+1].
    # -----------------------------------------------

    # pairIndex and signBit tables for r in [0..7].
    # r^7 is its complement in bits (since 7 = b111).
    # We'll define pairIndex(r) in {0..3} and signBit(r) in {+1, -1}.
    pair_idx = [0]*8
    sign_bit = [0]*8
    # Pairs: (0<->7), (1<->6), (2<->5), (3<->4)
    # We assign the smaller to +1, the bigger to -1.
    def build_pair_sign():
        for r in range(8):
            c = r ^ 7
            if r < c:
                # r is the "small" side
                sign_bit[r] = +1
            else:
                sign_bit[r] = -1
        # pair indices:
        # 0<->7 share pair 0
        # 1<->6 share pair 1
        # 2<->5 share pair 2
        # 3<->4 share pair 3
        # We'll hardcode or figure it out:
        # 0 -> pair 0, 7-> pair 0
        # 1 -> pair 1, 6-> pair 1
        # 2 -> pair 2, 5-> pair 2
        # 3 -> pair 3, 4-> pair 3
        pair_idx[0] = 0; pair_idx[7] = 0
        pair_idx[1] = 1; pair_idx[6] = 1
        pair_idx[2] = 2; pair_idx[5] = 2
        pair_idx[3] = 3; pair_idx[4] = 3

    build_pair_sign()

    # -----------------------------------------------
    # 3) DP structure:
    #    dp[i] will be a dictionary:
    #       key = (r, x0,x1,x2,x3, d0,d1,d2,d3)
    #       value = a list poly of length up to i*(i+1)//2 +1,
    #                poly[g] = number of ways (mod) for which we have
    #                exactly g good substrings so far, culminating in
    #                that state.
    #
    #    We'll build dp[i+1] from dp[i] by enumerating toggles[i].
    #
    #    Initially dp[0] has exactly 1 way: p(0)=0, freq(0)=1 => means
    #    x0=1 (the pair for 0 is pair 0), d0=+1, everything else 0.
    #    Good substrings so far = 0.
    # -----------------------------------------------

    from collections import defaultdict

    # A helper to create a zeroed polynomial array of length L.
    def make_poly(L):
        return [0]*(L)

    # Add src to dst (mod) (same length).
    def add_poly_inplace(dst, src):
        for i in range(len(dst)):
            val = dst[i]+src[i]
            if val>=mod: val-=mod
            dst[i]= val

    # Shift src by shift_ positions into new array (length old + shift).
    # We'll do it carefully to avoid re-allocation for each shift.
    def shift_poly(oldp, shift_):
        ln = len(oldp)
        newp = [0]*(ln+shift_)
        for i, val in enumerate(oldp):
            if val:
                newp[i+ shift_] = (newp[i+shift_] + val) % mod
        return newp

    # We store states as a tuple of 1 + 8 = 9 ints:
    # (r, x0,x1,x2,x3, d0,d1,d2,d3)

    # Construct dp for i=0
    # p(0)=0 => that means freq(0)=1 => pair(r=0)=0 => x0=1, d0=+1
    # all other pairs are x=0, d=0
    # # of good substrings=0 so far => poly: [1]
    init_state = (0, 1,0,0,0, 1,0,0,0)  # r=0, x0=1,x1=0,x2=0,x3=0, d0=1,d1=0,d2=0,d3=0
    dp_now = { init_state : [1] }  # polynomial of length 1 =>  dp_now[state][0] = 1

    # -----------------------------------------------
    # Iterate over each position i in [0..N-1]
    # Build dp_next from dp_now
    # -----------------------------------------------
    for i in range(N):
        dp_next = defaultdict(lambda: None)
        # The maximum # of good substrings so far can be i*(i+1)//2
        # For step i+1, it can go up to (i+1)*(i+2)//2
        new_poly_len = (i+1)*(i+2)//2 + 1

        # possible toggles for position i
        t_list = toggles[i]

        for state, poly in dp_now.items():
            # state = (r, x0,x1,x2,x3, d0,d1,d2,d3)
            (r, x0, x1, x2, x3, d0, d1, d2, d3) = state
            x_arr = [x0,x1,x2,x3]
            d_arr = [d0,d1,d2,d3]

            for t in t_list:
                r_new = r ^ t
                pidx = pair_idx[r_new]  # which pair
                # new good substrings = x_arr[pidx]
                cost = x_arr[pidx]

                # build the new state
                x_arr_new = x_arr[:]
                d_arr_new = d_arr[:]

                x_arr_new[pidx] = x_arr[pidx] + 1
                d_arr_new[pidx] = d_arr[pidx] + sign_bit[r_new]

                new_r = r_new
                new_state = (new_r,
                             x_arr_new[0], x_arr_new[1], x_arr_new[2], x_arr_new[3],
                             d_arr_new[0], d_arr_new[1], d_arr_new[2], d_arr_new[3])

                # shift the old polynomial by cost
                shifted = shift_poly(poly, cost)
                # now add into dp_next[new_state], which must be length new_poly_len
                oldp = dp_next[new_state]
                if oldp is None:
                    # create new
                    # we expect length up to new_poly_len
                    # shifted might be up to len(poly)+ cost,
                    # which is at most i*(i+1)//2 + cost. cost <= i+1 in worst case,
                    # so length <= i*(i+1)//2 + (i+1) ~ i^2 => fits in new_poly_len
                    L = max(new_poly_len, len(shifted))
                    newp = [0]*L
                    # copy shifted in
                    for idx, val in enumerate(shifted):
                        if val:
                            newp[idx] = (newp[idx] + val) % mod
                    dp_next[new_state] = newp
                else:
                    # add in
                    if len(shifted) > len(oldp):
                        # we need to extend oldp
                        old_len = len(oldp)
                        oldp.extend([0]*(len(shifted)-old_len))
                    for idx, val in enumerate(shifted):
                        if val:
                            s = oldp[idx] + val
                            if s>=mod: s-=mod
                            oldp[idx]= s

        dp_now = dp_next

    # -----------------------------------------------
    # Now dp_now = dp[N], each entry has a polynomial
    # of length up to N*(N+1)//2 +1 = 1276 for N=50.
    # We want sum of dp_now[state][g] for all g >= K.
    # -----------------------------------------------
    ans = 0
    upper_bound = (N*(N+1))//2  # maximum possible # good substrings
    for poly in dp_now.values():
        # sum poly[g] for g >= K
        # K can be up to N*(N+1)/2, so we just loop from K..len(poly)-1
        # or clamp if K >= len(poly).
        start = min(K, len(poly))
        for g in range(start, len(poly)):
            ans = (ans + poly[g]) % mod

    print(ans % mod)