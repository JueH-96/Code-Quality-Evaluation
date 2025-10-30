def main():
    import sys
    N, K = map(int, sys.stdin.readline().split())
    nk = N * K

    # Precompute factorials up to nk
    fact = [1] * (nk + 1)
    for i in range(1, nk + 1):
        fact[i] = fact[i-1] * i

    # Compute product: (fact[K])^N
    product = 1
    for _ in range(N):
        product *= fact[K]
        product = product  # Python handles big integers

    # Compute S and target
    S = fact[nk] // product
    target = (S + 1) // 2

    # Initialize counts and result
    counts = [K] * N
    result = []

    for position in range(nk):
        for x in range(1, N + 1):
            idx = x - 1
            if counts[idx] == 0:
                continue

            new_product = product // counts[idx]
            remaining = nk - position - 1

            if remaining < 0:
                multinom = 0
            else:
                multinom = fact[remaining] // new_product

            if multinom >= target:
                result.append(str(x))
                product = new_product
                counts[idx] -= 1
                break
            else:
                target -= multinom

    print(' '.join(result))

if __name__ == '__main__':
    main()