def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MAX_A = 10**5

    # Precompute smallest prime factor (spf) for each number up to MAX_A
    spf = [0]*(MAX_A+1)
    # Initialize spf with itself for each number
    for i in range(2, MAX_A+1):
        spf[i] = i

    # Sieve to fill smallest prime factors
    for i in range(2, int(MAX_A**0.5)+1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_A+1, i):
                if spf[j] == j:
                    spf[j] = i

    def grundy(x):
        # Calculate the sum of the exponents of prime factors for x
        # (that is, the total number of prime factors counting multiplicities)
        result = 0
        while x > 1:
            prime_factor = spf[x]
            result += 1
            x //= prime_factor
        return result

    nim = 0
    for num in A:
        nim ^= grundy(num)

    print("Anna" if nim != 0 else "Bruno")


# Do not forget to call main()!
if __name__ == "__main__":
    main()