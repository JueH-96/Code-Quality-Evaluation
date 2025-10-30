import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    data = sys.stdin
    M = 998244353
    line = data.readline().split()
    N = int(line[0]); Q = int(line[1])
    A_list = [0] + list(map(int, data.readline().split()))
    # precompute invs up to N
    inv = [0] * (N+1)
    inv[1] = 1
    for i in range(2, N+1):
        inv[i] = M - (M//i) * inv[M % i] % M
    # precompute C_i = 2*(i-1)/i^2 mod M, and build prefix sum of A_i * C_i
    prefC = [0] * (N+1)
    prefD = [0] * (N+1)  # for A_i * inv[i]
    for i in range(2, N+1):
        # C_i = 2*(i-1)/i^2 = 2*(i-1)*inv[i]*inv[i]
        ci = (2 * (i-1) % M) * inv[i] % M * inv[i] % M
        prefC[i] = (prefC[i-1] + A_list[i] * ci) % M
        prefD[i] = (prefD[i-1] + A_list[i] * inv[i]) % M
    # factorial (N-1)!
    fact = 1
    for i in range(1, N):
        fact = fact * i % M
    out = []
    for _ in range(Q):
        u,v = map(int, data.readline().split())
        # sum over i< u : prefC[u-1]
        if u-1 >= 2:
            sum1 = prefC[u-1]
        else:
            sum1 = 0
        # term_u: if u>=2: A_u*(u-1)/u = A_u - A_u*inv[u]
        if u >= 2:
            term_u = (A_list[u] - A_list[u] * inv[u]) % M
        else:
            term_u = 0
        # mid sum for i in [u+1 .. v-1]: A_i * (1/i)
        if v-1 >= max(u+1, 2):
            sum_mid = (prefD[v-1] - prefD[u]) % M
        else:
            sum_mid = 0
        # term_v: A_v * 1
        term_v = A_list[v] % M
        total = (sum1 + term_u + sum_mid + term_v) % M
        ans = total * fact % M
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()