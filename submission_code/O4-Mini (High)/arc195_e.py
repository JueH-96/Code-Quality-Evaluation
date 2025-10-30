import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin
    MOD = 998244353
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); Q = int(line[1])
    A = [0] * (N + 2)
    parts = data.readline().split()
    # A[2]..A[N]
    for i in range(2, N+1):
        A[i] = int(parts[i-2]) % MOD
    # precompute inverses up to N+1
    inv = [0] * (N + 3)
    inv[1] = 1
    for i in range(2, N+2):
        # inv[i] = pow(i, MOD-2, MOD)  # slower
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
    # precompute factorial up to N-1 to get T = (N-1)! mod M
    fact = [1] * (N + 1)
    for i in range(1, N):
        fact[i] = fact[i-1] * i % MOD
    T = fact[N-1]
    # precompute prefix sums Pre1 and Pre3
    # Pre1[i] = sum_{j=2..i} A[j] * C1[j], where C1[j] = 2*(j-1)/(j*(j+1))
    # Pre3[i] = sum_{j=2..i} A[j] * inv[j]
    Pre1 = [0] * (N + 2)
    Pre3 = [0] * (N + 2)
    # initialize
    Pre1[0] = 0
    Pre1[1] = 0
    Pre3[0] = 0
    Pre3[1] = 0
    for j in range(2, N+1):
        # C1 = 2*(j-1) * inv[j] * inv[j+1] mod
        # compute step by step
        c = (2 * (j-1)) % MOD
        c = c * inv[j] % MOD
        c = c * inv[j+1] % MOD
        Pre1[j] = Pre1[j-1] + A[j] * c % MOD
        if Pre1[j] >= MOD:
            Pre1[j] -= MOD
        # Pre3
        Pre3[j] = Pre3[j-1] + A[j] * inv[j] % MOD
        if Pre3[j] >= MOD:
            Pre3[j] -= MOD
    out = []
    for _ in range(Q):
        line = data.readline().split()
        u = int(line[0]); v = int(line[1])
        # S = sum_{i=2..u-1} A_i*C1[i] + A_u*(1 - 1/u) + sum_{i=u+1..v-1} A_i*(1/i) + A_v
        # using prefix sums:
        # Pre1[u-1] covers first sum
        # D_u = A_u*(1 - inv[u])
        # middle = Pre3[v-1] - Pre3[u]
        # last = A_v
        s = Pre1[u-1]
        # D_u
        # Note A[1]=0 so works for u=1
        Du = A[u] * (1 - inv[u]) % MOD
        s += Du
        if s >= MOD:
            s -= MOD
        # middle
        # v >= u+1 => v-1 >= u
        mid = Pre3[v-1] - Pre3[u]
        if mid < 0:
            mid += MOD
        s += mid
        if s >= MOD:
            s -= MOD
        # A_v
        s += A[v]
        if s >= MOD:
            s -= MOD
        # multiply by T
        ans = s * T % MOD
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()