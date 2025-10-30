MOD = 987644033

def main():
    import sys
    A, B = map(int, sys.stdin.readline().split())
    product = 1
    for i in range(1, A + 1):
        for j in range(1, B + 1):
            hook = (A - i + 1) + (B - j + 1) - 1
            product = (product * hook) % MOD

    # Compute (A*B)!
    n = A * B
    fact = 1
    for i in range(2, n + 1):
        fact = (fact * i) % MOD

    # Compute inverse of product modulo MOD
    inv_prod = pow(product, MOD - 2, MOD)

    result = (fact * inv_prod) % MOD
    result = (result * 2) % MOD

    print(result)

if __name__ == '__main__':
    main()