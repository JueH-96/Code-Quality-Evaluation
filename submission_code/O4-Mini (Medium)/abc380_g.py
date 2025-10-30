import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    P = [0] * (n+1)
    for i in range(1, n+1):
        P[i] = int(next(it))
    MOD = 998244353

    # Fenwick for counts (ints)
    class BITcnt:
        def __init__(self,n):
            self.n = n
            self.f = [0] * (n+1)
        def add(self,i,v):
            while i <= self.n:
                self.f[i] += v
                i += i & -i
        def sum(self,i):
            s = 0
            while i>0:
                s += self.f[i]
                i -= i & -i
            return s
        def range(self,l,r):
            if r<l: return 0
            return self.sum(r) - self.sum(l-1)

    # Fenwick for position sums mod
    class BITpos:
        def __init__(self,n):
            self.n = n
            self.f = [0] * (n+1)
        def add(self,i,v):
            # v is mod
            n = self.n
            f = self.f
            while i <= n:
                f[i] = (f[i] + v) % MOD
                i += i & -i
        def sum(self,i):
            s = 0
            f = self.f
            while i>0:
                s = (s + f[i]) % MOD
                i -= i & -i
            return s
        def range(self,l,r):
            if r<l: return 0
            return (self.sum(r) - self.sum(l-1)) % MOD

    # 1) compute initial inversions inv0
    bit0 = BITcnt(n)
    inv0 = 0
    for j in range(1, n+1):
        x = P[j]
        if x < n:
            inv0 += bit0.range(x+1, n)
        # else no greater
        bit0.add(x, 1)
    inv0 %= MOD

    # 2) compute T = sum_{i<j, P_i>P_j, j-i < K} (K - (j - i))
    bitc = BITcnt(n)
    bitp = BITpos(n)
    T = 0
    # Window BITs will hold P[L..j-1] where L = max(1, j-K+1)
    for j in range(1, n+1):
        x = P[j]
        # count of i in window with P_i > x
        if x < n:
            cnt_j = bitc.range(x+1, n)
            sumpos_j = bitp.range(x+1, n)
        else:
            cnt_j = 0
            sumpos_j = 0
        # T += cnt_j*(K - j) + sumpos_j
        # compute term mod
        term = (cnt_j * ((K - j) % MOD) + sumpos_j) % MOD
        T = (T + term) % MOD
        # add P[j] at pos j
        bitc.add(x, 1)
        bitp.add(x, j % MOD)
        # remove element that leaves window for next j+1
        rem = j - K + 1
        if rem >= 1:
            y = P[rem]
            bitc.add(y, -1)
            # remove position
            bitp.add(y, (-rem) % MOD)
    T %= MOD

    # compute final expected inv:
    # E = inv0 + K*(K-1)/4 - T/(n-K+1)  mod
    inv2 = (MOD + 1) // 2
    inv4 = inv2 * inv2 % MOD
    part_inside = K * (K-1) % MOD * inv4 % MOD
    inv_seg = pow(n - K + 1, MOD-2, MOD)
    res = (inv0 + part_inside - T * inv_seg) % MOD
    print(res)

if __name__ == "__main__":
    main()