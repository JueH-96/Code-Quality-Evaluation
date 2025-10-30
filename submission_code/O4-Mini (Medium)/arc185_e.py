import sys
import threading
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    MOD = 998244353
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    maxA = max(A)
    # compute phi up to maxA
    phi = list(range(maxA+1))
    for i in range(2, maxA+1):
        if phi[i] == i:
            step = i
            for j in range(i, maxA+1, step):
                phi[j] -= phi[j] // i
    # compute divisors for all numbers up to maxA
    divs = [[] for _ in range(maxA+1)]
    for d in range(1, maxA+1):
        for multiple in range(d, maxA+1, d):
            divs[multiple].append(d)
    # precompute powers of 2 mod MOD
    pow2 = [1] * (N+1)
    for i in range(1, N+1):
        pow2[i] = pow2[i-1] * 2 % MOD
    # f[t] = sum of pow2[i-1] over i<current where t divides A[i]
    f = [0] * (maxA+1)
    out = []
    F_prev = 0
    for m in range(1, N+1):
        x = A[m-1]
        S = 0
        # sum over divisors t of x: phi[t] * f[t]
        for t in divs[x]:
            S += phi[t] * f[t]
        S %= MOD
        F_cur = (2 * F_prev + S) % MOD
        out.append(str(F_cur))
        # update f[t] for next steps
        add = pow2[m-1]
        for t in divs[x]:
            f[t] = (f[t] + add) % MOD
        F_prev = F_cur
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()