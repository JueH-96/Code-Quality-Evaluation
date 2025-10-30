MOD = 998244353

def main():
    import sys
    n, k = map(int, sys.stdin.readline().split())
    
    if n == 1:
        print(1)
        return
        
    n_sq = pow(n, k, MOD)
    nm2 = n - 2
    if nm2 < 0:
        nm2 += MOD
    nm2_sq = pow(nm2, k, MOD)
    
    num = (n + 1) * n_sq - (n - 1) * nm2_sq
    num %= MOD
    den = 2 * n_sq % MOD
    inv_den = pow(den, MOD - 2, MOD)
    ans = num * inv_den % MOD
    print(ans)

if __name__ == "__main__":
    main()