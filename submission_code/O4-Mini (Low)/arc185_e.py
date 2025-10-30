import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MOD = 998244353
    maxA = max(A)

    # 1) Sieve for phi and spf
    phi = list(range(maxA+1))
    spf = [0] * (maxA+1)
    for i in range(2, maxA+1):
        if spf[i] == 0:
            spf[i] = i
            phi[i] = i-1
            for j in range(i*i, maxA+1, i):
                if spf[j] == 0:
                    spf[j] = i
                phi[j] = phi[j] // i * (i-1)
    # for primes > sqrt(maxA)
    for i in range(2, maxA+1):
        if spf[i] == i:
            # already done in loop
            continue
        if spf[i] == 0:
            spf[i] = i
            phi[i] = i-1

    # 2) precompute powers of 2 up to N
    pow2 = [1] * (N+1)
    for i in range(1, N+1):
        pow2[i] = (pow2[i-1] * 2) % MOD

    # 3) sumP[d] = sum of 2^{j-1} for prior indices j where A_j divisible by d
    sumP = [0] * (maxA+1)

    # helper: generate divisors of x via its prime factorization
    def gen_divisors(x):
        # factor x
        fac = []
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            fac.append((p, cnt))
        # produce divisors by recursion
        divs = [1]
        for p, cnt in fac:
            muls = []
            pp = 1
            for _ in range(cnt):
                pp *= p
                for d in divs:
                    muls.append(d * pp)
            divs += muls
        return divs

    out = []
    T_prev = 0
    # process each m
    for idx, val in enumerate(A, start=1):
        # S_m = sum_{d|val} phi[d]*sumP[d]
        S_m = 0
        # get divisors
        ds = gen_divisors(val)
        for d in ds:
            S_m += phi[d] * sumP[d]
        S_m %= MOD

        # update T_m = 2*T_{m-1} + S_m
        T = (T_prev * 2 + S_m) % MOD
        out.append(str(T))
        T_prev = T

        # update sumP for future: for each divisor d of val add 2^{idx-1}
        add = pow2[idx-1]
        for d in ds:
            sumP[d] = (sumP[d] + add) % MOD

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()