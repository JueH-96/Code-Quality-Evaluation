import sys

MOD = 998244353

def main():
    n = int(sys.stdin.readline())
    k = len(str(n))
    pow_10_k = pow(10, k, MOD)
    denominator = (pow_10_k - 1) % MOD
    
    if denominator == 0:
        sum_S_mod = n % MOD
    else:
        inv_denominator = pow(denominator, MOD - 2, MOD)
        e = k * n
        pow_10_e_mod = pow(10, e, MOD)
        numerator_mod = (pow_10_e_mod - 1) % MOD
        sum_S_mod = (numerator_mod * inv_denominator) % MOD
    
    ans = ( (n % MOD) * sum_S_mod ) % MOD
    print(ans)

if __name__ == "__main__":
    main()