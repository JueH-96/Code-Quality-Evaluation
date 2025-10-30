def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MAXA = 10**5

    # Build a sieve to find smallest prime factor (spf) for each number up to MAXA
    # spf[x] = smallest prime that divides x.
    spf = [0] * (MAXA + 1)
    for i in range(2, MAXA + 1):
        if spf[i] == 0:  # i is prime
            for j in range(i, MAXA + 1, i):
                if spf[j] == 0:
                    spf[j] = i

    # Function to compute the sum of prime exponents in factorization of x
    # Example: 72 = 2^3 * 3^2 => sum of exponents = 3 + 2 = 5
    def sum_of_exponents(x):
        total = 0
        while x > 1:
            total += 1
            x //= spf[x]
        return total

    # Compute the Nim-sum (xor of the Grundy values). G(n) = sum of exponents of n.
    nim_sum = 0
    for value in A:
        nim_sum ^= sum_of_exponents(value)

    # If the Nim-sum is non-zero, the first player (Anna) wins; otherwise Bruno wins.
    print("Anna" if nim_sum != 0 else "Bruno")

# Do not forget to call main function.
main()