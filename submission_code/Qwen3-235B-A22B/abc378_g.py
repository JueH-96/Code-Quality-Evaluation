import sys

def main():
    import sys
    A, B, M = map(int, sys.stdin.readline().split())
    n = A * B - 1
    if n <= 0:
        print(0)
        return

    # Precompute factorial mod M
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % M

    # Compute product of hook lengths
    product = 1
    for i in range(1, B + 1):
        if i < B:
            current_row = A
        else:
            current_row = A - 1
        for j in range(1, current_row + 1):
            # Compute 'below' for cell (i, j)
            below = 0
            for k in range(i + 1, B + 1):
                if k < B:
                    len_k = A
                else:
                    len_k = A - 1
                if len_k >= j:
                    below += 1
            # Hook length
            hook = (current_row - j) + below + 1
            product = product * hook % M

    # Compute inverse of product mod M
    if product == 0:
        print(0)
        return
    inv_product = pow(product, M - 2, M)

    # Compute f
    f = fact[n] * inv_product % M

    # Compute answer
    ans = f * (A - 1) % M
    ans = ans * (B - 1) % M
    print(ans)

if __name__ == "__main__":
    main()