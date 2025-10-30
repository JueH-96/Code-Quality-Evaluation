import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # Pre‐group by 2‐adic valuation k: A_i = 2^k * u (u odd)
    from collections import defaultdict
    groups = defaultdict(list)
    for a in A:
        k = (a & -a).bit_length() - 1
        u = a >> k
        groups[k].append(u)
    # Precompute cross‐group contributions for k<p
    total = 0
    ks = sorted(groups.keys())
    cnt = {k: len(groups[k]) for k in ks}
    sumu = {k: sum(groups[k]) for k in ks}
    for i, k in enumerate(ks):
        for p in ks[i+1:]:
            # pairs between group k and p (k<p)
            # for each u in k, v in p: f = u + v*2^(p-k)
            pw = 1 << (p - k)
            total += cnt[p] * sumu[k] + pw * cnt[k] * sumu[p]
    # Now same‐group: for each k, need sum_{i<=j} f(2^k (u_i+u_j))
    # f(2^k * s) = f(s).  And for diagonal i=j, s=2u_i => f(2u_i)=u_i.
    # Off‐diagonal i<j: need sum odd_part((u_i+u_j)//2)
    # We do naive O(m^2) if group small, else FFT convolution if large.
    def odd_part(x):
        return x >> ((x & -x).bit_length() - 1)
    for k, U in groups.items():
        m = len(U)
        if m <= 2000:
            # brute
            # diagonal
            for u in U:
                total += u
            # off‐diagonal
            for i in range(m):
                ui = U[i]
                for j in range(i+1, m):
                    s = ui + U[j]
                    total += odd_part(s)
        else:
            # FFT convolution approach
            # Build freq of u_i//2 (all u_i odd)
            maxu = max(U)
            size = (maxu//2) + 1
            # next power of two for FFT
            n = 1
            while n < 2*size: n <<= 1
            # real FFT convolution
            import numpy as np
            F = np.zeros(n, dtype=np.int64)
            for u in U:
                F[(u-1)//2] += 1
            # FFT via numpy
            FA = np.fft.rfft(F)
            C = np.fft.irfft(FA*FA, n).round().astype(np.int64)
            # C[t] = number of ordered pairs i,j with (u_i-1)//2 + (u_j-1)//2 = t
            # s = t+1.  Off‐diagonal unordered pairs: (C[t] - diag_t)//2
            # diag_t is count of i=j with 2*(u_i-1)//2 = t => u_i = 2*(t)//2+1 => t=(u_i-1)
            diag = np.zeros_like(C)
            for u in U:
                t = (u-1)//2 * 2
                # This entry counts the ordered pair (i,i) twice in C at index t//?
                # Actually for i=j, F[i]*F[i] contributes 1 at index 2*i
                diag[2*((u-1)//2)] += 1
            # sum diag contributions (each diag pair counted once)
            total += sum(U)
            # off‐diagonal
            for t in range(len(C)):
                s = t+1
                cnt_pairs = (C[t] - diag[t])//2
                if cnt_pairs:
                    total += cnt_pairs * odd_part(s)
    print(total)

if __name__ == "__main__":
    main()