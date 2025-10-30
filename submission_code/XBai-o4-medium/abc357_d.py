MOD = 998244353

def main():
    N = int(input())
    s = str(N)
    k = len(s)
    r = pow(10, k, MOD)
    if r == 1:
        sum_geo = N % MOD
    else:
        pow_r_N = pow(r, N, MOD)
        numerator = (pow_r_N - 1) % MOD
        denominator = (r - 1) % MOD
        inv_denominator = pow(denominator, MOD - 2, MOD)
        sum_geo = (numerator * inv_denominator) % MOD
    result = ( (N % MOD) * sum_geo ) % MOD
    print(result)

if __name__ == "__main__":
    main()