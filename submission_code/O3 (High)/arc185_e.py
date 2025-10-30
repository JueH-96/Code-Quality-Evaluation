import sys
sys.setrecursionlimit(1 << 25)

MOD = 998244353
MAXV = 100_000        # upper bound of A_i

# ------------------------------------------------------------
# pre–compute smallest prime factor (spf) and Euler totient
# ------------------------------------------------------------
spf = [0]*(MAXV+1)
primes = []
phi   = [0]*(MAXV+1)
phi[1] = 1
for i in range(2, MAXV+1):
    if spf[i] == 0:                       # i is prime
        spf[i] = i
        phi[i] = i-1
        primes.append(i)
    for p in primes:
        if p > spf[i] or i*p > MAXV:
            break
        spf[i*p] = p
        if i % p == 0:
            phi[i*p] = phi[i] * p
            break
        else:
            phi[i*p] = phi[i] * (p-1)

# ------------------------------------------------------------
# divisor enumeration (with memoisation)
# ------------------------------------------------------------
div_cache = {}
def divisors(x: int):
    if x in div_cache:
        return div_cache[x]
    org = x
    fact = []
    while x > 1:
        p = spf[x]
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        fact.append((p, cnt))
    divs = [1]
    for p, c in fact:
        cur = []
        mul = 1
        for _ in range(c):
            mul *= p
            for d in divs:
                cur.append(d*mul)
        divs += cur
    div_cache[org] = divs
    return divs

# ------------------------------------------------------------
def main() -> None:
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # pre-compute powers of two
    pow2 = [1]*(N+1)
    for i in range(1, N+1):
        pow2[i] = (pow2[i-1] << 1) % MOD   # (2*prev) % MOD

    # F[d] = sum of pow2[pos-1] for previous positions with value divisible by d
    F = [0]*(MAXV+1)

    S = 0          # answer for current prefix
    out_lines = []
    for idx, val in enumerate(A, 1):        # idx is 1-based position m
        divs = divisors(val)

        # R = Σ_{d|val} φ(d) * F[d]
        R = 0
        for d in divs:
            R += phi[d] * F[d]
        R %= MOD

        # update total answer for current prefix
        S = (2*S + R) % MOD
        out_lines.append(str(S))

        # add current element's weight to F
        w = pow2[idx-1]            # 2^{idx-1}
        for d in divs:
            F[d] = (F[d] + w) % MOD

    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()