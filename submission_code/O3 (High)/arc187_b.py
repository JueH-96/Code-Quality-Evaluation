import sys

MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    B = [int(next(it)) for _ in range(N)]

    INF = M + 1
    # prefix data
    pref_unknown = [0] * N           # unknowns in 0..i
    pref_min     = [INF] * N         # minimal fixed value in 0..i
    unk = 0
    cur_min = INF
    for i in range(N):
        if B[i] == -1:
            unk += 1
        else:
            cur_min = min(cur_min, B[i])
        pref_unknown[i] = unk
        pref_min[i] = cur_min

    # suffix data
    suff_unknown = [0] * (N + 1)     # unknowns in i..N-1, suff_unknown[N]=0
    suff_max     = [0] * (N + 1)     # maximal fixed value in i..N-1
    unk = 0
    cur_max = 0
    for i in range(N - 1, -1, -1):
        if B[i] == -1:
            unk += 1
        else:
            cur_max = max(cur_max, B[i])
        suff_unknown[i] = unk
        suff_max[i] = cur_max

    total_unknown = pref_unknown[-1]

    # base term: every assignment contributes at least 1 component
    ans = pow(M, total_unknown, MOD)

    q = total_unknown
    # power array for t-1 (initially t=0, so 0^k)
    pow_low_prev = [1] + [0]*q  # 0^0 =1, 0^k=0 for k>0

    for t in range(1, M):                     # suffix max (exact) = t
        low_val   = t
        upper_val = M - t                     # values allowed in prefix

        # powers of t  and M-t  up to exponent q
        pow_low  = [1]*(q+1)                  # t^k
        pow_up   = [1]*(q+1)                  # (M-t)^k
        for k in range(1, q+1):
            pow_low[k] = pow_low[k-1] * low_val % MOD
            pow_up[k]  = pow_up[k-1]  * upper_val % MOD

        # traverse boundaries
        for i in range(N-1):                  # boundary after index i
            # data for this boundary
            pref_min_val = pref_min[i]
            if pref_min_val <= t:             # prefix contains fixed value ≤ t → impossible
                continue
            suff_max_val = suff_max[i+1]
            if suff_max_val > t:              # suffix fixed value  > t  → impossible
                continue

            ku = pref_unknown[i]              # unknowns in prefix
            ks = suff_unknown[i+1]            # unknowns in suffix

            # prefix count
            prefix_cnt = pow_up[ku]           # (M-t)^{ku}

            # suffix count where exact max = t
            if suff_max_val == t:
                suffix_cnt = pow_low[ks]      # already has fixed t, only need ≤ t
            else:  # suff_max_val < t
                suffix_cnt = (pow_low[ks] - pow_low_prev[ks]) % MOD  # at least one t
            contribution = prefix_cnt * suffix_cnt % MOD
            ans = (ans + contribution) % MOD

        # advance prev array
        pow_low_prev = pow_low

    print(ans % MOD)

if __name__ == "__main__":
    main()