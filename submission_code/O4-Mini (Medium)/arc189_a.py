import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    mod = 998244353
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # initial X[i] = i%2 (1-based)
    # compute segments of equal A
    segs = []
    i = 0
    while i < N:
        j = i+1
        while j < N and A[j]==A[i]:
            j+=1
        segs.append((i, j-1, j-i))  # (left_idx, right_idx, length)
        i = j

    # Pre-check: segments of length 1 must match initial parity
    for (l, r, L) in segs:
        if L == 1:
            # position l (0-based) corresponds to cell idx = l+1
            if A[l] != ((l+1)&1):
                print(0)
                return

    # For each segment length L, require L odd
    ks = []
    total_k = 0
    for (l, r, L) in segs:
        if L & 1 == 0:
            print(0)
            return
        k = (L - 1) >> 1
        ks.append(k)
        total_k += k

    # Precompute factorials and inv factorials up to max N
    M = N  # up to N is enough since 2*k_j <= N
    fact = [1] * (M+1)
    for i in range(1, M+1):
        fact[i] = fact[i-1] * i % mod
    invfact = [1] * (M+1)
    invfact[M] = pow(fact[M], mod-2, mod)
    for i in range(M, 0, -1):
        invfact[i-1] = invfact[i] * i % mod

    inv2 = (mod+1)//2

    # compute product of C(2k, k) over segments
    prodC = 1
    for k in ks:
        if k > 0:
            c = fact[2*k] * invfact[k] % mod * invfact[k] % mod
            prodC = (prodC * c) % mod
        # if k==0, C(0,0)=1 no effect

    # compute inv2^total_k
    pow_inv2 = pow(inv2, total_k, mod)

    # number of interleavings = total_k! / prod(k_j!)
    # but we incorporate that via fact[total_k] later.
    # ans = fact[total_k] * prod( C(2k,k) * inv2^k ) 
    #     = fact[total_k] * prodC * inv2^{total_k}
    ans = fact[total_k] * prodC % mod * pow_inv2 % mod

    print(ans)

if __name__ == "__main__":
    main()