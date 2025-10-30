mod = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    inv6 = pow(6, mod-2, mod)
    a = N % mod
    b = M % mod
    c = (b + 1) % mod
    term = (3 * a * b) % mod
    term = (term - (2 * b % mod)) % mod
    term = (term + 2) % mod
    numerator = a * b % mod
    numerator = numerator * c % mod
    numerator = numerator * term % mod
    result = numerator * inv6 % mod
    print(result)

if __name__ == "__main__":
    main()