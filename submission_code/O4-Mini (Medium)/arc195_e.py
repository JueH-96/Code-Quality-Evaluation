import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    it = iter(data)
    mod = 998244353
    N = int(next(it))
    Q = int(next(it))
    A = [0] * (N + 1)
    for i in range(2, N + 1):
        A[i] = int(next(it)) % mod
    # precompute inverses up to N+1
    inv = [0] * (N + 2)
    inv[1] = 1
    for i in range(2, N + 2):
        inv[i] = mod - (mod // i) * inv[mod % i] % mod
    # compute (N-1)! mod
    fact = 1
    for i in range(1, N):
        fact = fact * i % mod
    # precompute C1, C2, C3 and their prefix sums P1, P3
    # C1[i] = 2/(i+1) * A[i]
    # C2[i] = (i-1)/i * A[i]
    # C3[i] = 1/i * A[i]
    C1 = [0] * (N + 1)
    C2 = [0] * (N + 1)
    C3 = [0] * (N + 1)
    for i in range(2, N + 1):
        # inv[i], inv[i+1] available
        C1[i] = A[i] * 2 % mod * inv[i+1] % mod
        C2[i] = A[i] * (i - 1) % mod * inv[i] % mod
        C3[i] = A[i] * inv[i] % mod
    # prefix sums
    P1 = [0] * (N + 1)
    P3 = [0] * (N + 1)
    for i in range(2, N + 1):
        P1[i] = (P1[i-1] + C1[i]) % mod
        P3[i] = (P3[i-1] + C3[i]) % mod
    out = []
    for _ in range(Q):
        u = int(next(it))
        v = int(next(it))
        # u < v guaranteed
        # sum = sum_{i< u} 2/(i+1)*A[i]  +  (u-1)/u*A[u]
        #     + sum_{u< i< v} 1/i * A[i]  + A[v]
        s = 0
        # part1: i=2..u-1
        if u-1 >= 2:
            s = (s + P1[u-1]) % mod
        # part2: i=u
        # C2[u] valid for u>=2; for u=1, C2[1]=0
        s = (s + C2[u]) % mod
        # part3: i=u+1..v-1
        # v-1 >= u always since u<v
        # P3[v-1] - P3[u]
        if v - 1 >= u + 1:
            s = (s + P3[v-1] - P3[u]) % mod
        # part4: i=v
        s = (s + A[v]) % mod
        if s < 0:
            s += mod
        ans = s * fact % mod
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()