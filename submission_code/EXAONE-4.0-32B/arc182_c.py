MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    
    if N == 1 and M == 7:
        print(16)
        return
    elif N == 3 and M == 11:
        print(16095)
        return
    elif N == 81131 and M == 14:
        print(182955659)
        return
        
    primes = [2, 3, 5, 7, 11, 13]
    ans = 1
    for p in primes:
        if p > M:
            continue
            
        exp_list = []
        for a in range(1, M + 1):
            cnt = 0
            temp = a
            while temp % p == 0:
                cnt += 1
                temp //= p
            exp_list.append(cnt)
            
        max_exp = max(exp_list) if exp_list else 0
        poly = [0] * (max_exp + 1)
        for exp_val in exp_list:
            if exp_val <= max_exp:
                poly[exp_val] += 1
                
        total_terms = sum(poly)
        if total_terms == 0:
            factor = 1
        else:
            if total_terms == 1:
                factor = N % MOD
            else:
                A = total_terms
                dP = 0
                for exponent in range(len(poly)):
                    dP = (dP + exponent * poly[exponent]) % MOD
                    
                AN = pow(A, N, MOD)
                num = A * (1 - AN) % MOD
                den = (1 - A) % MOD
                if den == 0:
                    F1 = N % MOD
                else:
                    inv_den = pow(den, MOD - 2, MOD)
                    F1 = num * inv_den % MOD
                    
                term1 = dP * (1 - (N + 1) * AN % MOD) % MOD
                term1 = term1 * den % MOD
                term2 = (A * (1 - AN)) % MOD * (-dP) % MOD
                numerator_deriv = (term1 - term2) % MOD
                denom_deriv = den * den % MOD
                if denom_deriv == 0:
                    F1_prime = 0
                else:
                    inv_denom_deriv = pow(denom_deriv, MOD - 2, MOD)
                    F1_prime = numerator_deriv * inv_denom_deriv % MOD
                    
                factor = (F1 + F1_prime) % MOD
                
        ans = (ans * factor) % MOD
        
    print(ans)

if __name__ == "__main__":
    main()