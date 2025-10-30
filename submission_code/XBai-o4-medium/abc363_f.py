import math

def has_no_zero(x):
    return '0' not in str(x)

def is_perfect_square(x):
    s = math.isqrt(x)
    return s * s == x

def split_factors(X):
    if X == 1:
        return [1]
    if has_no_zero(X):
        return [X]
    for d in range(2, int(math.isqrt(X)) + 1):
        if X % d == 0:
            # Try d
            if has_no_zero(d):
                rest = split_factors(X // d)
                if rest is not None:
                    return [d] + rest
            # Try X//d
            e = X // d
            if has_no_zero(e):
                rest = split_factors(d)
                if rest is not None:
                    return [e] + rest
    return None

def get_divisors(n):
    divisors = set()
    divisors.add(1)
    divisors.add(n)
    i = 2
    while i * i <= n:
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
        i += 1
    return divisors

def main():
    N = int(input().strip())
    s_N = str(N)
    if s_N == s_N[::-1] and '0' not in s_N:
        print(s_N)
        return
    # Check even case
    if is_perfect_square(N):
        sqrt_N = math.isqrt(N)
        factors = split_factors(sqrt_N)
        if factors is not None:
            parts = [str(x) for x in factors]
            s = '*'.join(parts + parts[::-1])
            print(s)
            return
    # Check odd case
    divisors = get_divisors(N)
    for m in divisors:
        if not has_no_zero(m):
            continue
        if N % m != 0:
            continue
        rem = N // m
        if is_perfect_square(rem):
            x = math.isqrt(rem)
            factors_x = split_factors(x)
            if factors_x is not None:
                parts = [str(f) for f in factors_x]
                middle = str(m)
                s = '*'.join(parts + [middle] + parts[::-1])
                print(s)
                return
    print(-1)

if __name__ == '__main__':
    main()