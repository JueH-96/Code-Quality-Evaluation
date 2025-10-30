import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    B = [0]*(n+1)
    for i in range(1, n+1):
        B[i] = int(next(it))
    mod = 998244353

    # Precompute factorials and inverse factorials up to 2*n
    N = 2*n + 5
    fac = [1]* (N)
    invfac = [1]* (N)
    for i in range(1, N):
        fac[i] = fac[i-1] * i % mod
    # Fermat inverse factorial
    invfac[N-1] = pow(fac[N-1], mod-2, mod)
    for i in range(N-1, 0, -1):
        invfac[i-1] = invfac[i] * i % mod

    def C(x, y):
        # binomial x choose y mod
        if y < 0 or y > x or x < 0:
            return 0
        return fac[x] * (invfac[y] * invfac[x-y] % mod) % mod

    # Compute avail slots after each prefix of B
    # avail[0] = 1
    avail = [0] * (n+1)
    avail[0] = 1
    for i in range(1, n+1):
        # consume one slot, add B[i] new slots
        avail[i] = avail[i-1] - 1 + B[i]

    # Check if B prefix[1..j] valid for j=1..n-1
    # i.e., avail[j] >= 1 for j < n
    valid_prefix = True
    for j in range(1, n):
        if avail[j] < 1:
            valid_prefix = False
            break

    ans = 0
    # Count lexicographically smaller sequences
    # Loop i from 1..n
    for i in range(1, n+1):
        x = avail[i-1]
        if x < 1:
            # no further prefixes possible
            break
        Bi = B[i]
        if i < n:
            # positions where we choose S[i] < B[i]
            if Bi <= 0:
                continue
            # rem nodes after position i
            m = n - i
            # k_min = max(1, x-1)
            if x > 1:
                k_min = x - 1
            else:
                k_min = 1
            # k_max0 = x + Bi - 2
            k_max0 = x + Bi - 2
            # k_max cannot exceed rem
            if k_max0 < 1:
                continue
            if k_max0 > m:
                k_max = m
            else:
                k_max = k_max0
            if k_min > k_max:
                continue
            # compute t_min, t_max
            # t = 2*m - k
            t_min = 2*m - k_max
            t_max = 2*m - k_min
            # sum_{k=k_min..k_max} a(m,k)
            # a(m,k) = 2*C(t-1, m-1) - C(t,m) with t=2m-k
            # sum = 2*(C(t_max, m) - C(t_min-1, m)) - (C(t_max+1, m+1) - C(t_min, m+1))
            # compute components
            # C(t_max, m)
            c_tm = C(t_max, m)
            # C(t_min-1, m)
            c_tmin1 = C(t_min - 1, m)
            # C(t_max+1, m+1)
            c_tm1 = C(t_max + 1, m + 1)
            # C(t_min, m+1)
            c_tmin = C(t_min, m + 1)
            # assemble
            # 2*(c_tm - c_tmin1) - (c_tm1 - c_tmin)
            part = (c_tm - c_tmin1) % mod
            part = (part * 2) % mod
            diff = (c_tm1 - c_tmin) % mod
            cnt = (part - diff) % mod
            ans = (ans + cnt) % mod
        else:
            # i == n, last position: choose v < B[n] s.t. final avail becomes 0
            # avail_n = x -1 + v == 0 => v = 1 - x
            # requires v>=0 and v < B[n]
            v = 1 - x
            if 0 <= v < Bi:
                ans += 1
                if ans >= mod:
                    ans -= mod

    # Finally include equal sequence B if it is Polish
    # B is Polish if valid_prefix and avail[n] == 0
    if valid_prefix and avail[n] == 0:
        ans = (ans + 1) % mod

    print(ans)

if __name__ == "__main__":
    main()