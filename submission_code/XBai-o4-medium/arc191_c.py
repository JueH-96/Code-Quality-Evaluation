import sys

def is_prime(n):
    if n <= 1:
        return False
    elif n <= 3:
        return True
    elif n % 2 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 3, 5, 7, 11]
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s-1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def get_unique_prime_factors(x):
    factors = set()
    if x % 2 == 0:
        factors.add(2)
        while x % 2 == 0:
            x //= 2
    i = 3
    while i * i <= x:
        if x % i == 0:
            factors.add(i)
            while x % i == 0:
                x //= i
        i += 2
    if x > 1:
        factors.add(x)
    return factors

def find_primitive_root(m, phi_m, factors):
    candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for g in candidates:
        if g >= m:
            continue
        is_primitive = True
        for q in factors:
            if pow(g, phi_m // q, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    for g in range(2, m):
        if g in candidates:
            continue
        is_primitive = True
        for q in factors:
            if pow(g, phi_m // q, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return None

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    cases = list(map(int, input[1:T+1]))
    for N in cases:
        if N == 1:
            print("2 1")
        elif N <= 60:
            m = (1 << N) - 1
            print(f"2 {m}")
        else:
            k = 1
            while True:
                m_candidate = k * N + 1
                if is_prime(m_candidate):
                    break
                else:
                    k += 1
            m = m_candidate
            phi_m = k * N
            factors = get_unique_prime_factors(phi_m)
            g = find_primitive_root(m, phi_m, factors)
            A = pow(g, k, m)
            print(f"{A} {m}")

if __name__ == "__main__":
    main()