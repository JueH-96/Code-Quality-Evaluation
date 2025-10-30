import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    mod = 998244353
    
    def is_good(num):
        if num <= 0:
            return False
        sum_divisors = 1
        temp_num = num
        i = 2
        while i * i <= temp_num:
            if temp_num % i == 0:
                power = 0
                while temp_num % i == 0:
                    temp_num //= i
                    power += 1
                sum_divisors *= (pow(i, power + 1, mod * 3) - 1) * pow(i - 1, -1, mod * 3)
                sum_divisors %= (mod * 3)
            i += 1
        if temp_num > 1:
            sum_divisors *= (pow(temp_num, 1 + 1, mod * 3) - 1) * pow(temp_num - 1, -1, mod * 3)
            sum_divisors %= (mod * 3)
        return sum_divisors % 3 == 0

    if m == 1:
        count = 0
        for i in range(1, min(n, 1000) + 1):
            if is_good(i):
                count += 1
        if n <= 1000:
            print(count % mod)
            return
        
    primes = []
    is_prime_flag = [True] * (300)
    is_prime_flag[0] = is_prime_flag[1] = False
    for p in range(2, 300):
        if is_prime_flag[p]:
            primes.append(p)
            for i in range(p * p, 300, p):
                is_prime_flag[i] = False
                
    q1_primes = []
    q2_primes = []
    for p in primes:
        if p % 3 == 1:
            q1_primes.append(p)
        elif p % 3 == 2:
            q2_primes.append(p)
            
    v_n = {}
    temp_n = n
    for p in primes:
        if p > n:
            break
        count = 0
        while temp_n % p == 0:
            temp_n //= p
            count += 1
        v_n[p] = count
    if temp_n > 1:
        v_n[temp_n] = 1
        
    prime_factors_n = list(v_n.keys())
    prime_factors_n.sort()
    
    s_values = {}
    
    for p in prime_factors_n:
        vp = v_n[p]
        if p == 3:
            s_values[p] = (pow(vp + m + 1, m, mod) - pow(vp + m, m, mod)) * pow(pow(m, m, mod), mod - 2, mod) % mod
            if s_values[p] < 0:
                s_values[p] += mod
        elif p in q2_primes:
            s_p = 0
            for k in range(vp // 2 + 1):
                num = 2 * k + m - 1
                den = m - 1
                if num < den:
                    continue
                nCr = 1
                for i in range(den):
                    nCr = (nCr * (num - i)) % mod
                    nCr = (nCr * pow(i + 1, mod - 2, mod)) % mod
                s_p = (s_p + nCr) % mod
            s_values[p] = s_p
        elif p in q1_primes:
            s_p = 0
            for k in range(vp // 3 + 1):
                num = 3 * k + m - 1
                den = m - 1
                if num < den:
                    continue
                nCr = 1
                for i in range(den):
                    nCr = (nCr * (num - i)) % mod
                    nCr = (nCr * pow(i + 1, mod - 2, mod)) % mod
                s_p = (s_p + nCr) % mod
            for k in range((vp - 1) // 3 + 1):
                num = 3 * k + m
                den = m - 1
                if num < den:
                    continue
                nCr = 1
                for i in range(den):
                    nCr = (nCr * (num - i)) % mod
                    nCr = (nCr * pow(i + 1, mod - 2, mod)) % mod
                s_p = (s_p + nCr) % mod
            s_values[p] = s_p
        else:
            s_values[p] = 1
            
    not_good_sequences_count = 1
    for p in prime_factors_n:
        not_good_sequences_count = (not_good_sequences_count * s_values[p]) % mod
        
    total_sequences_count_poly = [1]
    one_minus_x_inv_m = [0] * (n + 1)
    one_minus_x_inv_m[0] = 1
    
    for i in range(1, n + 1):
        term = 1
        for j in range(m):
            term = (term * (i + j)) % mod
        term = (term * pow(factorial(m), mod - 2, mod)) % mod
        one_minus_x_inv_m[i] = term
        
    total_sequences_count = 0
    for i in range(1, n + 1):
        coeff = 1
        if i >= m:
            num = i - 1
            den = m - 1
            if num >= den:
                nCr = 1
                for j in range(den):
                    nCr = (nCr * (num - j)) % mod
                    nCr = (nCr * pow(j + 1, mod - 2, mod)) % mod
                coeff = nCr
            else:
                coeff = 0
        else:
            coeff = 0
        if i == 1:
            coeff = 1
        total_sequences_count = (total_sequences_count + coeff) % mod

    def nCr_mod_p(n, r, p):
        if r < 0 or r > n:
            return 0
        num = 1
        den = 1
        for i in range(r):
            num = (num * (n - i)) % p
            den = (den * (i + 1)) % p
        return (num * pow(den, p - 2, p)) % p

    def get_total_sequences_count(n, m, mod):
        return nCr_mod_p(n + m, m, mod)
        
    total_sequences_count = get_total_sequences_count(n, m, mod)

    not_good_count = not_good_sequences_count
    good_count = (total_sequences_count - not_good_count + mod) % mod

    print(good_count)

def factorial(n):
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

if __name__ == '__main__':
    solve()