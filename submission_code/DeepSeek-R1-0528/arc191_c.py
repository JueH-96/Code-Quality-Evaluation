import math
import sys

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes

primes_35000 = sieve(35000)

def factorize(n):
    factors = []
    temp = n
    for p in primes_35000:
        if p * p > temp:
            break
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp //= p
            factors.append((p, cnt))
    if temp > 1:
        factors.append((temp, 1))
    return factors

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n < 35000:
        return n in set(primes_35000)
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def solve_CRT(equations):
    total = 1
    for m, _ in equations:
        total *= m
    A = 0
    for m, a in equations:
        Mi = total // m
        inv = pow(Mi, -1, m)
        A = (A + a * Mi * inv) % total
    if A == 0:
        A = total
    return A

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        N = int(data[index])
        index += 1
        if N == 1:
            out_lines.append("20250126 1")
            continue
        fac = factorize(N)
        M_val = 1
        crt_list = []
        use_alternative = False
        for (p, exp) in fac:
            if p == 2:
                if exp == 1:
                    m_i = 3
                    a_i = 2
                else:
                    m_i = 2 ** (exp + 2)
                    a_i = 5
            else:
                k = 1
                while True:
                    m_i = k * (p ** exp) + 1
                    if is_prime(m_i):
                        a_i = pow(2, k, m_i)
                        break
                    k += 1
            if M_val > (10 ** 18) // m_i:
                use_alternative = True
                break
            M_val *= m_i
            crt_list.append((m_i, a_i))
        if not use_alternative:
            A = solve_CRT(crt_list)
            out_lines.append(f"{A} {M_val}")
        else:
            if N % 2 == 0:
                k0 = 1
                step = 1
            else:
                k0 = 2
                step = 2
            k = k0
            found = False
            while k <= 1000000:
                p_val = k * N + 1
                if p_val > 10 ** 18:
                    break
                if is_prime(p_val):
                    valid = True
                    for (p_i, exp) in fac:
                        d = N // p_i
                        if pow(2, k * d, p_val) == 1:
                            valid = False
                            break
                    if valid:
                        A_val = pow(2, k, p_val)
                        out_lines.append(f"{A_val} {p_val}")
                        found = True
                        break
                k += step
            if found:
                continue
            else:
                M_val = 1
                crt_list = []
                for (p, exp) in fac:
                    if p == 2:
                        if exp == 1:
                            m_i = 3
                            a_i = 2
                        else:
                            m_i = 2 ** (exp + 2)
                            a_i = 5
                    else:
                        k_val = 1
                        while True:
                            m_i = k_val * (p ** exp) + 1
                            if is_prime(m_i):
                                a_i = pow(2, k_val, m_i)
                                break
                            k_val += 1
                    M_val *= m_i
                    crt_list.append((m_i, a_i))
                A = solve_CRT(crt_list)
                if 1 <= A <= 10 ** 18 and 1 <= M_val <= 10 ** 18:
                    out_lines.append(f"{A} {M_val}")
                else:
                    out_lines.append("20250126 1")
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()