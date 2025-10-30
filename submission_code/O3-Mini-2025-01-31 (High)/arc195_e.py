# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    mod = 998244353
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    q = int(next(it))
    # We use 1-indexing.
    # A[1] is not used (set to 0) and input gives A2, A3, …, A_n.
    A = [0]*(n+1)
    for i in range(2, n+1):
        A[i] = int(next(it)) % mod

    # Precompute inverses for 1..(n+2)
    inv = [0]*(n+3)
    inv[1] = 1
    for i in range(2, n+3):
        inv[i] = mod - (mod // i) * inv[mod % i] % mod

    # Precompute prefix P:
    # Let P[i] = sum_{j=2}^{i} (A[j] * inv(j)) for i>=2, with P[1]=0.
    P = [0]*(n+1)
    P[1] = 0
    for i in range(2, n+1):
        P[i] = (P[i-1] + A[i] * inv[i]) % mod

    # Now define X(x):
    # X(1) = 0 and for x>=2, X(x) = A[x] + P[x-1].
    X = [0]*(n+1)
    X[1] = 0
    for x in range(2, n+1):
        X[x] = (A[x] + P[x-1]) % mod

    # Precompute L array:
    # L(1) = 0. For x>=2, L(x) = sum_{i=2}^{x-1} X(i)*(inv(i)-inv(i+1)) + X(x)*inv(x).
    # We may compute L recursively:
    L_arr = [0]*(n+1)
    L_arr[1] = 0
    if n >= 2:
        L_arr[2] = X[2] * inv[2] % mod
    for x in range(2, n):
        diff = (X[x+1] - X[x]) % mod
        L_arr[x+1] = (L_arr[x] + diff * inv[x+1]) % mod

    # Precompute factorial (n-1)! mod mod.
    fact = 1
    for i in range(1, n):
        fact = fact * i % mod

    out_lines = []
    for _ in range(q):
        u = int(next(it))
        v = int(next(it))
        # Answer = fact * ( X(u) + X(v) - 2 * L(u) ) mod mod.
        res = fact * ( (X[u] + X[v] - 2 * L_arr[u]) % mod ) % mod
        out_lines.append(str(res))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()