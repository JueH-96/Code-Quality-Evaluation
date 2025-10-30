import sys

MOD = 998244353

def build_inverses(n):
    inv = [0]*(n+2)
    inv[1] = 1
    for i in range(2, n+2):
        inv[i] = MOD - (MOD//i) * inv[MOD % i] % MOD
    return inv

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    N = int(next(it))
    Q = int(next(it))

    A = [0]*(N+1)          # A[0],A[1] unused (A_1 does not exist)
    for i in range(2, N+1):
        A[i] = int(next(it)) % MOD

    # Pre–compute modular inverses up to N+1
    inv = build_inverses(N+1)

    # Pre–compute prefix sums needed for answering queries
    pre1 = [0]*(N+1)   # Σ_{k=2..i} A_k * 2(k-1)/(k(k+1))
    pre3 = [0]*(N+1)   # Σ_{k=2..i} A_k / k
    for i in range(2, N+1):
        # c1 = 2*(i-1)/(i*(i+1))
        c1 = (2*(i-1) % MOD) * inv[i] % MOD * inv[i+1] % MOD
        pre1[i] = (pre1[i-1] + A[i] * c1) % MOD

        # c3 = 1/i
        c3 = inv[i]
        pre3[i] = (pre3[i-1] + A[i] * c3) % MOD

    # carry over prefixes for i =1 (they stay 0)
    pre1[1] = pre1[0]
    pre3[1] = pre3[0]

    # (N-1)!
    fact = 1
    for i in range(2, N):
        fact = fact * i % MOD

    out_lines = []
    for _ in range(Q):
        u = int(next(it))
        v = int(next(it))
        # Guaranteed u < v

        res = 0

        if u > 1:
            # part from i = 2 .. u-1 ; coefficient c1
            res += pre1[u-1]

            # i = u ; coefficient (u-1)/u
            cu = (u-1) * inv[u] % MOD
            res += A[u] * cu % MOD

        # i between u+1 and v-1 ; coefficient 1/i
        diff = (pre3[v-1] - pre3[u]) % MOD
        res = (res + diff) % MOD

        # i = v ; coefficient 1
        res = (res + A[v]) % MOD

        answer = fact * res % MOD
        out_lines.append(str(answer))

    sys.stdout.write('
'.join(out_lines))

if __name__ == '__main__':
    main()