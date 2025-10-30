MOD = 998244353
import sys

def main():
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    L = 1 << 20
    freq = [0] * L
    for a in A:
        freq[a] += 1

    arr = freq[:]
    n_val = L
    m = 1
    while m < n_val:
        for i in range(0, n_val, 2 * m):
            for j in range(i, i + m):
                x = arr[j]
                y = arr[j + m]
                arr[j] = x + y
                arr[j + m] = x - y
        m <<= 1
    F_arr = arr

    D1 = [None] * (N + 1)
    D2 = [None] * (N + 1)
    
    cur1 = [0] * M
    cur1[0] = 1
    D1[0] = cur1[:]
    
    cur2 = [0] * M
    cur2[0] = 1
    D2[0] = cur2[:]
    
    for n in range(1, N + 1):
        nxt1 = [0] * M
        for r in range(M):
            idx_prev = (r - 1) % M
            nxt1[r] = (cur1[r] + cur1[idx_prev]) % MOD
        cur1 = nxt1
        D1[n] = cur1[:]
        
    for n in range(1, N + 1):
        nxt2 = [0] * M
        for r in range(M):
            idx_prev = (r - 1) % M
            nxt2[r] = (cur2[r] - cur2[idx_prev]) % MOD
        cur2 = nxt2
        D2[n] = cur2[:]
        
    H0 = [0] * L
    for k in range(L):
        total_val = N + F_arr[k]
        n_plus = total_val // 2
        n_minus = N - n_plus
        V1 = D1[n_plus]
        V2 = D2[n_minus]
        s = 0
        for i in range(M):
            j = (M - i) % M
            s = (s + V1[i] * V2[j]) % MOD
        H0[k] = s

    n_val = L
    m = 1
    while m < n_val:
        for i in range(0, n_val, 2 * m):
            for j in range(i, i + m):
                x = H0[j]
                y = H0[j + m]
                H0[j] = (x + y) % MOD
                H0[j + m] = (x - y) % MOD
        m <<= 1
        
    invL = pow(L, MOD - 2, MOD)
    for x in range(L):
        H0[x] = (H0[x] * invL) % MOD
        
    ans = 0
    for x in range(L):
        if H0[x] == 0:
            continue
        if x == 0:
            term = 0
        else:
            term = pow(x, K, MOD)
        ans = (ans + H0[x] * term) % MOD
        
    ans %= MOD
    print(ans)

if __name__ == "__main__":
    main()