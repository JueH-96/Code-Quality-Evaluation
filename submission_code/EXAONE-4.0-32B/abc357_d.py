MOD = 998244353

def main():
    import sys
    N_val = int(sys.stdin.readline())
    if N_val == 0:
        print(0)
        return
        
    d = len(str(N_val))
    r = pow(10, d, MOD)
    n_mod = N_val % MOD
    
    if r == 1:
        result = (n_mod * n_mod) % MOD
    else:
        T = pow(r, N_val, MOD) - 1
        if T < 0:
            T += MOD
        denom = (r - 1) % MOD
        inv_denom = pow(denom, MOD - 2, MOD)
        S = T * inv_denom % MOD
        result = n_mod * S % MOD
        
    print(result)

if __name__ == "__main__":
    main()