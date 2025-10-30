import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1])
    A = [0]*N
    for i in range(N):
        parts = data.readline().split()
        v = 0
        # bit 0 is A_{*,1}, bit M-1 is A_{*,M}
        # we pack little-endian: bit k-1 -> bit (k-1)
        for k,ch in enumerate(parts):
            if ch=='1':
                v |= 1<<k
        A[i] = v
    # mask of M bits
    mask = (1<<M) - 1
    # prepare shifts for prefix XOR: ks = [1,2,4,... <M]
    ks = []
    k = 1
    while k < M:
        ks.append(k)
        k <<= 1
    # build map from bitmask to list of indices
    mp = {}
    for idx,v in enumerate(A):
        mp.setdefault(v, []).append(idx)
    # period = smallest power of two >= M
    K = 1
    while (1<<K) < M:
        K += 1
    period = 1<<K
    mod = 998244353
    ans = 0
    # function to apply one T step: prefix xor
    def Tstep(x):
        # compute prefix xor of x within M bits
        for sh in ks:
            x ^= (x << sh) & mask
        return x
    # for each i, simulate T^x(A[i]) for x = 1..period
    for i in range(N):
        cur = A[i]
        # we only count j>=i, so compare j>i
        # run through one full period
        for x in range(1, period+1):
            # apply T once
            # prefix xor
            for sh in ks:
                cur ^= (cur << sh) & mask
            # lookup matches
            lst = mp.get(cur)
            if lst:
                # for each index j with same pattern, if j>=i, add x
                # note f(i,i)=0, so skip j==i
                for j in lst:
                    if j > i:
                        ans += x
                        if ans >= mod:
                            ans %= mod
        # end of period; next i
    print(ans % mod)

if __name__ == "__main__":
    main()