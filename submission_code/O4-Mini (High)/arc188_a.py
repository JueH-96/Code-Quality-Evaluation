def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    S = next(it)
    MOD = 998244353

    # Quick cases
    Q = S.count('?')
    # If no good‐substring threshold, all replacements count:
    if K == 0:
        print(pow(3, Q, MOD))
        return
    # If no '?' at all, just check the single fixed string
    if Q == 0:
        # compute parity‐prefix visits
        pair_id = [min(s, 7 - s) for s in range(8)]
        u = [0,0,0,0]
        x = 0
        # P[0]=0
        u[pair_id[0]] += 1
        for ch in S:
            if   ch == 'A': x ^= 1
            elif ch == 'B': x ^= 2
            else:            x ^= 4
            u[pair_id[x]] += 1
        f = 0
        for v in u:
            f += v*(v-1)//2
        print(1 if f >= K else 0)
        return

    # DP on the 4 complement‐pairs of parity states:
    # id 0 = {0,7}, id 1 = {1,6}, id 2 = {2,5}, id 3 = {3,4}
    # precompute transition: id → new_id under A,B,C
    idtrans = [
        [1, 2, 3],  # from id0: A→id1, B→id2, C→id3
        [0, 3, 2],  # from id1: A→id0, B→id3, C→id2
        [3, 0, 1],  # from id2: A→id3, B→id0, C→id1
        [2, 1, 0],  # from id3: A→id2, B→id1, C→id0
    ]
    # each time we visit group id we do u_id +=1
    # we encode (u0,u1,u2,u3) in one int:
    #   code = (u0<<12) | (u1<<6) | (u2)
    # and u3 = total_prefixes - (u0+u1+u2).
    # when we increment u_id, that adds 1<<shift_bits:
    idshift = [1<<12, 1<<6, 1, 0]

    char2idx = {'A':0, 'B':1, 'C':2}

    # dp_cur[id] = dict mapping code→ways
    dp_cur = [dict() for _ in range(4)]
    dp_next = [dict() for _ in range(4)]
    # initial prefix P[0]=0 has id=0 → u0=1
    init_code = 1 << 12
    dp_cur[0][init_code] = 1

    for ch in S:
        # clear next
        for i in range(4):
            dp_next[i].clear()
        if ch == '?':
            # branch on A,B,C
            for d in range(4):
                cd = dp_cur[d]
                if not cd:
                    continue
                row = idtrans[d]
                # for each letter prefetch new_id, shift
                nid0, sh0 = row[0], idshift[row[0]]
                nid1, sh1 = row[1], idshift[row[1]]
                nid2, sh2 = row[2], idshift[row[2]]
                nxt0 = dp_next[nid0]
                nxt1 = dp_next[nid1]
                nxt2 = dp_next[nid2]
                for code, ways in cd.items():
                    c0 = code + sh0
                    prev = nxt0.get(c0)
                    if prev is None:
                        nxt0[c0] = ways
                    else:
                        v = prev + ways
                        if v >= MOD: v -= MOD
                        nxt0[c0] = v
                    c1 = code + sh1
                    prev = nxt1.get(c1)
                    if prev is None:
                        nxt1[c1] = ways
                    else:
                        v = prev + ways
                        if v >= MOD: v -= MOD
                        nxt1[c1] = v
                    c2 = code + sh2
                    prev = nxt2.get(c2)
                    if prev is None:
                        nxt2[c2] = ways
                    else:
                        v = prev + ways
                        if v >= MOD: v -= MOD
                        nxt2[c2] = v
        else:
            # fixed letter
            li = char2idx[ch]
            for d in range(4):
                cd = dp_cur[d]
                if not cd:
                    continue
                nd = idtrans[d][li]
                sh = idshift[nd]
                nxt = dp_next[nd]
                for code, ways in cd.items():
                    c1 = code + sh
                    prev = nxt.get(c1)
                    if prev is None:
                        nxt[c1] = ways
                    else:
                        v = prev + ways
                        if v >= MOD: v -= MOD
                        nxt[c1] = v
        dp_cur, dp_next = dp_next, dp_cur

    # final tally
    total = N + 1
    ans = 0
    for d in range(4):
        for code, ways in dp_cur[d].items():
            u0 = (code >> 12) & 0x3F
            u1 = (code >> 6) & 0x3F
            u2 = code & 0x3F
            u3 = total - (u0 + u1 + u2)
            f = (u0*(u0-1) + u1*(u1-1) +
                 u2*(u2-1) + u3*(u3-1)) // 2
            if f >= K:
                ans += ways
                if ans >= MOD:
                    ans -= MOD
    print(ans)

if __name__ == '__main__':
    main()