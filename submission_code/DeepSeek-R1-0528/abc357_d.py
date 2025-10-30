MOD = 998244353

def main():
    import sys
    N = int(sys.stdin.readline())
    d = len(str(N))
    exponent = d * N
    base_val = pow(10, d, MOD)
    denom = (base_val - 1) % MOD

    if denom == 0:
        S = N % MOD
    else:
        num_val = pow(10, exponent, MOD)
        num = (num_val - 1) % MOD
        inv_denom = pow(denom, MOD - 2, MOD)
        S = num * inv_denom % MOD
    
    result = (N % MOD) * S % MOD
    print(result)

if __name__ == '__main__':
    main()