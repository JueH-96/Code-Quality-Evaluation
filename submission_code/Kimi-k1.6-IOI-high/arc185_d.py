MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    
    nm = (N % MOD) * (M % MOD) % MOD
    term_part = (nm - M + 1) % MOD
    term1 = (6 * term_part) % MOD
    
    three_nm = (3 * nm) % MOD
    minus_two_m = (-2 * M) % MOD
    plus_four = 4 % MOD
    part = (three_nm + minus_two_m + plus_four) % MOD
    
    multiplier = (M - 1) % MOD
    term2 = (multiplier * part) % MOD
    
    sum_terms = (term1 + term2) % MOD
    numerator = (nm * sum_terms) % MOD
    
    inv_6 = pow(6, MOD-2, MOD)
    ans = (numerator * inv_6) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()