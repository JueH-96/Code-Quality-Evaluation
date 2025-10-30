import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Compute runs: lengths and start positions (0-based)
    runs_len = []
    runs_start = []
    i = 0
    while i < N:
        s = i
        # extend while same
        while i+1 < N and A[i+1] == A[i]:
            i += 1
        # run from s..i
        runs_start.append(s)
        runs_len.append(i - s + 1)
        i += 1
    m = len(runs_len)
    # Validity checks
    k = 0
    k_j = []
    for idx in range(m):
        s = runs_start[idx]
        length = runs_len[idx]
        # length must be odd
        if length % 2 == 0:
            print(0)
            return
        # run start parity must match initial X
        # initial X at position s (0-based) is (s+1)%2
        if A[s] != ((s+1) & 1):
            print(0)
            return
        kj = (length - 1) // 2
        k_j.append(kj)
        k += kj
    # Precompute factorials up to max needed = max(N, 2*k)
    mod = 998244353
    max_n = max(N, 2*k, k)
    # We can bound max_n by N (since 2*k = sum(len_j -1) <= N)
    max_n = N
    fac = [1] * (max_n + 1)
    ifac = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fac[i] = fac[i-1] * i % mod
    # Fermat inverse factorial
    ifac[max_n] = pow(fac[max_n], mod-2, mod)
    for i in range(max_n, 0, -1):
        ifac[i-1] = ifac[i] * i % mod
    # Compute answer
    ans = fac[k]
    # multiply by product of C(2*k_j, k_j)
    for kj in k_j:
        # C(2*kj, kj) = fac[2*kj] / (fac[kj]*fac[kj])
        c = fac[2*kj] * ifac[kj] % mod * ifac[kj] % mod
        ans = ans * c % mod
    # multiply by inverse of 2^k
    # inv2 = (mod+1)//2
    inv2 = (mod + 1) // 2
    ans = ans * pow(inv2, k, mod) % mod
    print(ans)

if __name__ == "__main__":
    main()