import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    K = int(next(it))
    # 1-based permutation
    P = [0] * (n + 1)
    for i in range(1, n + 1):
        P[i] = int(next(it))
    M = 998244353

    # --- 1) compute initial inversion count Inv0 using a Fenwick tree (bit1) ---
    bit1 = [0] * (n + 1)
    inv0 = 0
    for idx in range(1, n + 1):
        x = P[idx]
        # sum_leq = bit1.sum(x)
        s = 0
        t = x
        while t > 0:
            s += bit1[t]
            t -= t & -t
        # seen so far = idx-1, so greater = (idx-1) - s
        inv0 += (idx - 1) - s
        # bit1.add(x,1)
        t = x
        while t <= n:
            bit1[t] += 1
            t += t & -t
    inv0_mod = inv0 % M

    # --- 2) compute S_in = sum over s of inv_inside(s),
    #     sliding window of length K, maintain inversion count (inv_curr) in window ---
    bit2 = [0] * (n + 1)
    ws = 0       # current window size in bit2
    inv_curr = 0
    # build initial window [1..K]
    for i in range(1, K + 1):
        x = P[i]
        # count <= x in window so far
        s = 0
        t = x
        while t > 0:
            s += bit2[t]
            t -= t & -t
        # count > x = ws - s
        inv_curr += (ws - s)
        # insert x
        t = x
        while t <= n:
            bit2[t] += 1
            t += t & -t
        ws += 1

    S_in = inv_curr
    tot_s = n - K + 1
    # slide window from s=2 to s=tot_s
    # window at step s contains positions [s..s+K-1]
    for s_idx in range(2, tot_s + 1):
        # remove element P[s_idx-1]
        x = P[s_idx - 1]
        # count of values < x in current window
        sl = 0
        t = x - 1
        while t > 0:
            sl += bit2[t]
            t -= t & -t
        # those sl pairs were inversions where x > something to its right
        inv_curr -= sl
        # actually remove x from bit2
        t = x
        while t <= n:
            bit2[t] -= 1
            t += t & -t
        ws -= 1

        # add new element P[s_idx+K-1]
        y = P[s_idx + K - 1]
        # count <= y in window
        sy = 0
        t = y
        while t > 0:
            sy += bit2[t]
            t -= t & -t
        # count > y = ws - sy
        inv_curr += (ws - sy)
        # insert y
        t = y
        while t <= n:
            bit2[t] += 1
            t += t & -t
        ws += 1

        S_in += inv_curr

    # --- 3) compute final expected value mod M ---
    # E = Inv0 - (S_in)/(tot_s) + K*(K-1)/4
    invT = pow(tot_s, M-2, M)
    inv4 = pow(4, M-2, M)
    res = (inv0_mod - (S_in % M) * invT + (K * (K - 1) % M) * inv4) % M
    print(res)

if __name__ == "__main__":
    main()